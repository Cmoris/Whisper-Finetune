"""Offline checks: python -m unittest discover -s tests -p test_train_distil.py."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainingArguments, WhisperConfig, WhisperForConditionalGeneration
from transformers.utils import is_accelerate_available

from model.transformer import TransformerConfig, TransformerForConditionalGeneration
from train_distil import DistillationTrainer, distillation_loss, parse_args


class DistillationTests(unittest.TestCase):
    def test_mask_temperature_and_teacher_gradients(self):
        student = torch.randn(2, 3, 11, requires_grad=True)
        teacher = torch.randn(2, 3, 11, requires_grad=True)
        labels = torch.tensor([[1, 2, -100], [3, -100, -100]])
        mask = labels.ne(-100)
        expected = F.kl_div(F.log_softmax(student[mask] / 2, -1),
                            F.softmax(teacher[mask].detach() / 2, -1),
                            reduction="batchmean") * 4
        actual = distillation_loss(student, teacher, labels, 2)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[~mask].abs().sum().item(), 0)
        empty = distillation_loss(student, teacher, torch.full_like(labels, -100))
        self.assertEqual(empty.item(), 0)
        empty.backward()

    def make_models(self):
        torch.set_num_threads(1)
        common = dict(vocab_size=16, num_mel_bins=4, d_model=8,
                      encoder_layers=1, decoder_layers=1,
                      encoder_attention_heads=2, decoder_attention_heads=2,
                      encoder_ffn_dim=16, decoder_ffn_dim=16,
                      max_source_positions=4, max_target_positions=8,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2,
                      decoder_start_token_id=1)
        student = TransformerForConditionalGeneration(TransformerConfig(**common))
        teacher = WhisperForConditionalGeneration(WhisperConfig(**common))
        return student, teacher

    def test_seq2seq_loss_and_backward(self):
        student, teacher = self.make_models()
        teacher.requires_grad_(False).eval()
        batch = dict(input_features=torch.randn(2, 4, 8),
                     labels=torch.tensor([[3, 2, -100], [4, 5, 2]]))
        context = SimpleNamespace(teacher_model=teacher, loss_lambda=0.3, temperature=1.5)
        loss, outputs = DistillationTrainer.compute_loss(context, student, batch, return_outputs=True)
        with torch.no_grad():
            expected_kd = distillation_loss(outputs.logits, teacher(**batch).logits, batch["labels"])
        expected_ce = F.cross_entropy(outputs.logits.float().reshape(-1, 16),
                                      batch["labels"].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(loss, 0.3 * expected_ce + 0.7 * expected_kd)
        loss.backward()
        self.assertTrue(torch.isfinite(student.lm_head.weight.grad).all())
        self.assertGreater(student.lm_head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

        for weight in (0.0, 1.0):
            context.loss_lambda = weight
            value, result = DistillationTrainer.compute_loss(context, student, batch, return_outputs=True)
            expected = (result.loss if weight == 1 else
                        distillation_loss(result.logits, teacher(**batch).logits, batch["labels"]))
            torch.testing.assert_close(value, expected)
        batch["labels"].fill_(-100)
        empty = DistillationTrainer.compute_loss(context, student, batch)
        self.assertEqual(empty.item(), 0)
        empty.backward()

    def test_lambda_cli(self):
        for option in ("--lambda", "--loss_lambda", "--alpha"):
            self.assertEqual(parse_args([option, "0.3"]).loss_lambda, 0.3)

    @unittest.skipUnless(is_accelerate_available(), "Trainer integration requires accelerate")
    def test_training_evaluation_save_and_resume(self):
        student, teacher = self.make_models()
        data = [dict(input_features=torch.randn(4, 8), labels=torch.tensor([3, 2, -100]))
                for _ in range(4)]
        workspace_output = Path(__file__).resolve().parents[1] / "output"
        workspace_output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_output) as output_dir:
            args = Seq2SeqTrainingArguments(
                output_dir=output_dir, use_cpu=True, report_to="none", max_steps=1,
                per_device_train_batch_size=1, gradient_accumulation_steps=2,
                per_device_eval_batch_size=2, save_steps=1,
                prediction_loss_only=True, disable_tqdm=True,
                label_names=["labels"], remove_unused_columns=False,
            )
            trainer = DistillationTrainer(model=student, teacher_model=teacher,
                                          args=args, train_dataset=data, eval_dataset=data)
            before = student.lm_head.weight.detach().clone()
            trainer.train()
            self.assertFalse(torch.equal(before, student.lm_head.weight))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
            self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"])))
            checkpoint = output_dir + "/checkpoint-1"
            loaded = TransformerForConditionalGeneration.from_pretrained(checkpoint)
            torch.testing.assert_close(loaded.lm_head.weight, student.lm_head.weight)
            args.max_steps = 2
            resumed = DistillationTrainer(model=loaded, teacher_model=teacher,
                                          args=args, train_dataset=data)
            resumed.train(resume_from_checkpoint=checkpoint)
            self.assertEqual(resumed.state.global_step, 2)


if __name__ == "__main__":
    unittest.main()

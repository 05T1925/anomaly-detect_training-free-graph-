import torch

from model import ARCZero
from utils import test_eval


class ARCZeroTrainer:
    def __init__(self, train_config, model_config, data):
        self.train_config = train_config
        self.model_config = model_config
        self.data = data
        self.model = ARCZero(**model_config).to(train_config["device"])

    def train_and_evaluate(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config["lr"],
            weight_decay=self.model_config["weight_decay"],
        )

        for _ in range(self.train_config["epochs"]):
            for train_data in self.data["train"]:
                self.model.train()
                train_graph = train_data.graph.to(self.train_config["device"])
                residual_embed = self.model(train_graph)
                loss = self.model.cross_attn.get_train_loss(
                    residual_embed,
                    train_graph.ano_labels,
                    self.model_config["num_prompt"],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()
        test_scores = {}

        for i, test_data in enumerate(self.data["test"]):
            test_graph = test_data.graph.to(self.train_config["device"])
            labels = test_graph.ano_labels
            shot_mask = test_graph.shot_mask.bool()
            residual_embed = self.model(test_graph)

            t = self.model_config["t"]
            k = self.model_config["k"]

            with torch.no_grad():
                sum_scores = torch.zeros_like(labels, dtype=torch.float)
                count = 0

                query_scores = self.model.cross_attn.get_test_score(residual_embed, shot_mask)
                min_score = torch.min(query_scores)
                all_scores = torch.zeros_like(labels, dtype=torch.float)
                all_scores[~shot_mask] = query_scores
                all_scores[shot_mask] = min_score
                sum_scores += all_scores
                count += 1

                for _ in range(t):
                    query_indices = torch.nonzero(~shot_mask).squeeze(1)
                    _, lowest_k_indices = torch.topk(query_scores, k, largest=False)
                    shot_mask = torch.zeros_like(shot_mask)
                    shot_mask[query_indices[lowest_k_indices]] = True

                    query_scores = self.model.cross_attn.get_test_score(residual_embed, shot_mask)
                    min_score = torch.min(query_scores)
                    all_scores = torch.zeros_like(labels, dtype=torch.float)
                    all_scores[~shot_mask] = query_scores
                    all_scores[shot_mask] = min_score
                    sum_scores += all_scores
                    count += 1

            final_scores = sum_scores / count
            metrics = test_eval(labels.to(self.train_config["device"]), final_scores)

            test_name = self.train_config["testdsets"][i]
            test_scores[test_name] = {
                "AUROC": metrics["AUROC"],
                "AUPRC": metrics["AUPRC"],
            }

        return test_scores

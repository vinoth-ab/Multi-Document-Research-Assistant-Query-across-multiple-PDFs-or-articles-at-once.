"""
Utility script to generate realistic sample scientific research papers for testing
the Multi-Document Research Assistant.
"""

from pathlib import Path
import pymupdf as fitz


def create_paper_pdf(filepath: Path, title: str, pages_data: list):
    """Generate a multi-page PDF formatted as an academic paper."""
    doc = fitz.open()

    for page_idx, page_content in enumerate(pages_data):
        page = doc.new_page(width=595, height=842)  # Standard A4 size
        # Header
        page.insert_text(
            (50, 45),
            f"Academic Research Publication — {title}",
            fontsize=8,
            color=(0.4, 0.4, 0.4),
        )
        page.insert_text(
            (520, 45),
            f"Page {page_idx + 1}",
            fontsize=8,
            color=(0.4, 0.4, 0.4),
        )
        page.draw_line((50, 52), (545, 52), color=(0.7, 0.7, 0.7), width=0.5)

        # Title on page 1
        y_cursor = 75
        if page_idx == 0:
            page.insert_text(
                (50, y_cursor),
                title,
                fontsize=15,
                color=(0.1, 0.1, 0.3),
            )
            y_cursor += 25
            page.insert_text(
                (50, y_cursor),
                "Authors: Dr. Elena Vance, Prof. Marcus Chen, Dr. Sarah Al-Mansoor",
                fontsize=9,
                color=(0.3, 0.3, 0.3),
            )
            y_cursor += 15
            page.insert_text(
                (50, y_cursor),
                "Institute of Advanced Artificial Intelligence & Data Science Research",
                fontsize=8,
                color=(0.4, 0.4, 0.4),
            )
            y_cursor += 25

        # Content blocks
        for section_heading, body_paragraphs in page_content:
            if section_heading:
                page.insert_text(
                    (50, y_cursor),
                    section_heading,
                    fontsize=11,
                    color=(0.15, 0.25, 0.45),
                )
                y_cursor += 18

            for para in body_paragraphs:
                # Wrap text roughly inside margin
                rect = fitz.Rect(50, y_cursor, 545, y_cursor + 90)
                rc = page.insert_textbox(rect, para, fontsize=9.5, lineheight=1.35)
                # Advance cursor based on inserted text height
                y_cursor += (len(para) // 65 + 1) * 14 + 10

        # Footer
        page.draw_line((50, 805), (545, 805), color=(0.8, 0.8, 0.8), width=0.5)
        page.insert_text(
            (50, 818),
            "Confidential Peer-Reviewed Pre-Print — For Academic & Research Evaluation Only",
            fontsize=7.5,
            color=(0.5, 0.5, 0.5),
        )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(filepath))
    doc.close()
    print(f"Generated: {filepath} ({len(pages_data)} pages)")


def generate_all_samples():
    sample_dir = Path("./sample_docs")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paper 1: NLP Transformer Architecture
    paper1_pages = [
        [
            ("1. Abstract & Introduction", [
                "Modern Natural Language Processing relies on Transformer architectures utilizing multi-head self-attention mechanisms. In this study, we propose a sparse attention adaptation called SparseSelfAttn that reduces quadratic computational complexity O(N^2) to near-linear O(N sqrt(N)).",
                "Our experiments on the GLUE benchmark achieve an overall score of 89.4%, outperforming standard BERT-Base while utilizing 38% less GPU memory during training.",
            ]),
            ("2. Model Architecture & Theoretical Formulation", [
                "The core self-attention formulation computes attention weights across query, key, and value vectors: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. Standard models compute attention across all token pairs.",
                "In SparseSelfAttn, attention connections are restricted to sliding local windows of size k=64 tokens combined with 16 global landmark tokens, preserving both contextual nuance and global document semantics.",
            ]),
        ],
        [
            ("3. Empirical Evaluation & Benchmarks", [
                "We evaluated our architecture on MNLI, QQP, SST-2, and SQuAD v2.0 datasets. On MNLI-matched accuracy, the model achieved 86.8%, representing a 1.2% improvement over baseline architectures.",
                "Training was conducted on a cluster of 8 NVIDIA A100 GPUs (80GB VRAM) for 500,000 steps using AdamW optimizer with learning rate 3e-4, linear warmup for 10,000 steps, and cosine decay.",
            ]),
            ("4. Limitations & Future Directions", [
                "While memory efficiency is substantially improved, our implementation currently lacks specialized kernel optimization for non-tensor core architectures.",
                "Furthermore, extreme document lengths (>16,384 tokens) still exhibit slight perplexity degradation on long-form narrative coherence benchmarks.",
            ]),
        ],
    ]
    create_paper_pdf(sample_dir / "Sparse_Transformer_NLP.pdf", "SparseSelfAttn: Efficient Linear-Complexity Transformers for NLP", paper1_pages)

    # 2. Paper 2: Computer Vision ResNet vs ViT
    paper2_pages = [
        [
            ("1. Abstract & Research Motivation", [
                "Convolutional Neural Networks (CNNs) have historically dominated visual recognition tasks due to their translational invariance and local receptive fields. Recently, Vision Transformers (ViT) have challenged this dominance by treating image patches as sequence tokens.",
                "In this paper, we conduct an extensive empirical study comparing ResNet-50 and ViT-B/16 across ImageNet-1K classification, robustness to out-of-distribution noise, and inference throughput on edge devices.",
            ]),
            ("2. Experimental Setup and Hyperparameters", [
                "Both models were trained using RandAugment data augmentation, CutMix, and Mixup over 300 epochs. Optimization used SGD with momentum 0.9 for ResNet, and AdamW with weight decay 0.05 for ViT.",
                "Images were resized to 224x224 pixels with 16x16 patch projection for ViT-B/16. Total parameter count is 25.6M parameters for ResNet-50 compared to 86.4M parameters for ViT-B/16.",
            ]),
        ],
        [
            ("3. Comparative Results & Accuracy", [
                "On ImageNet-1K top-1 accuracy, ViT-B/16 achieved 81.2% while ResNet-50 achieved 79.1%. However, on corrupted ImageNet-C benchmarks, ViT demonstrated higher resilience with 14% lower error rate under severe adversarial Gaussian noise.",
                "Conversely, inference speed on edge platforms (NVIDIA Jetson AGX Orin) favored ResNet-50, which executed at 142 frames per second (FPS), compared to 54 FPS for ViT-B/16 due to attention matrix latency.",
            ]),
            ("4. Conclusions and Open Challenges", [
                "Our findings indicate that while Vision Transformers scale superiorly with massive pre-training data, CNNs remain optimal for constrained hardware deployments.",
                "Future research should explore hybrid CNN-Transformer architectures that balance inductive bias with self-attention flexibility.",
            ]),
        ],
    ]
    create_paper_pdf(sample_dir / "Vision_Transformers_vs_CNNs.pdf", "Empirical Evaluation of Vision Transformers and ResNets in Computer Vision", paper2_pages)

    # 3. Paper 3: Federated Learning & Privacy in Healthcare
    paper3_pages = [
        [
            ("1. Abstract & Executive Summary", [
                "Clinical artificial intelligence deployment faces critical data governance hurdles due to patient privacy laws including HIPAA and GDPR. Centralized data pooling from disparate hospital networks introduces severe legal liabilities and data leak vulnerabilities.",
                "This paper presents HealthFed-DP, a privacy-preserving federated learning framework for training deep diagnostic classifiers on multi-site chest X-ray radiographs without sharing patient raw data.",
            ]),
            ("2. Privacy Guarantees & Differential Privacy", [
                "HealthFed-DP incorporates Renyi Differential Privacy (RDP) with parameter epsilon = 1.25 and delta = 1e-5. Gradient clipping threshold C was fixed at 1.0, and Gaussian noise was injected into client weight updates before server aggregation.",
                "Secure multi-party aggregation was executed using homomorphic encryption (Paillier cryptosystem), ensuring the central server cannot inspect individual hospital model updates.",
            ]),
        ],
        [
            ("3. Multi-Institutional Clinical Results", [
                "We deployed HealthFed-DP across 5 simulated hospital nodes comprising 42,000 chest radiography scans for pneumonia and cardiomegaly detection. The federated model achieved an Area Under the ROC Curve (AUC) of 0.932.",
                "In comparison, a model trained on individual hospital data achieved only 0.814 AUC, demonstrating the immense statistical value of collaborative multi-institutional training without raw data migration.",
            ]),
            ("4. Challenges and Regulatory Limitations", [
                "Network bandwidth constraints during homomorphic encryption aggregation introduced a 2.4x latency increase over unencrypted federated baselines.",
                "Furthermore, non-IID data distribution across specialized regional clinics caused minor convergence instability in rare disease classes.",
            ]),
        ],
    ]
    create_paper_pdf(sample_dir / "Federated_Learning_Healthcare_Privacy.pdf", "HealthFed-DP: Differential Privacy and Federated Learning in Healthcare", paper3_pages)


if __name__ == "__main__":
    generate_all_samples()

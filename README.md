# FALCON-Net-Feature-Aggregation-of-Local-Patterns-for-AI-Generated-Image-Detection

# FALCON-Net: Feature Aggregation for Localized Context and Noise Network

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

## 📖 Abstract

With the rapid development of generative models, the visual quality of generated images has become almost indistinguishable from real images, which poses a huge challenge to content authenticity verification. Existing methods for detecting AI-generated images mainly rely on specific forgery cues, which are usually tailored to specific generative models and have difficulty in achieving good generalization across different diverse models and data distributions.

Based on the observation of local differences in the generated images, we found that the generated images lack device-specific sensor noise and unnatural pixel intensity variations caused by the oversimplified generation process. These discrepancies provide important forensic cues for distinguishing between real and generated images.

We propose the **Feature Aggregation for Localized Context and Noise Network (FALCON-Net)**, which leverages these discrepancies to enhance detection capabilities. FALCON-Net integrates two complementary modules:

- **Intrinsic Noise Pattern Isolation (INP) module**: Captures the difference between generated images and real images in sensor fingerprints by frequency domain analysis of high-frequency noise features.
- **Local Variation Pattern (LVP) module**: Reveals unnatural regularities in generated images by modeling the complex relationship between local pixels.

By leveraging the complementary strengths of the INP and LVP modules, FALCON-Net extracts critical forensic cues at both the sensor and local structural levels. The INP module isolates device-specific noise patterns through frequency domain analysis, while the LVP module captures directional pixel intensity variations to detect unnatural regularities. This design enables FALCON-Net to identify fundamental generation inconsistencies, ensuring robustness to post-processing operations and strong generalization to diverse and unseen generative models.

Extensive experimental results show that FALCON-Net achieves the state-of-the-art performance in detecting generated images and shows good generalization ability to unseen generative models.

## 🏗️ Architecture Overview

![FALCON-Net Architecture](assets/overview.png)

*FALCON-Net Architecture: Dual-branch feature extraction with INP (Intrinsic Noise Pattern) and LVP (Local Variation Pattern) modules, followed by feature fusion and classification through a pruned ResNet backbone.*

FALCON-Net processes input images through two parallel feature extraction branches before concatenating their outputs and feeding them into a pruned ResNet classifier:

### 1. **Local Directional Pattern (LVP) Extraction Branch**
- **Pixel Window Processing**: Analyzes 3×3 pixel neighborhoods to capture local intensity relationships
- **Direction Encoding**: Converts pixel intensity differences into directional codes
- **Pattern Generation**: Produces grayscale LVP maps representing local directional features

### 2. **Image Noise Pattern (INP) / Frequency Domain Feature Extraction Branch**
- **Frequency Transform**: Applies FFT to obtain raw frequency spectrum
- **High-Frequency Masking**: Removes low-frequency components to focus on noise patterns
- **Feature Enhancement**: Amplifies high-frequency features for better detection
- **IFFT Reconstruction**: Converts enhanced frequency features back to spatial domain

### 3. **Feature Fusion and Classification**
- **Concatenation**: Combines LVP and INP feature maps
- **Pruned ResNet**: Deep neural network with residual connections for feature processing
- **Binary Classification**: Softmax output distinguishing between "REAL" and "FAKE" images

## ✨ Key Features

- **Dual-Module Design**: Combines sensor-level (INP) and structural-level (LVP) analysis
- **Frequency Domain Analysis**: Leverages FFT/IFFT for noise pattern isolation
- **Local Pixel Modeling**: Captures unnatural regularities in generated images
- **Robust Generalization**: Strong performance across diverse generative models
- **Post-Processing Resistance**: Maintains detection accuracy under various manipulations

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (for GPU acceleration)
- Other dependencies (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/FALCON-Net.git
cd FALCON-Net

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
import torch
from falcon_net import FALCONNet

# Initialize model
model = FALCONNet()

# Load pre-trained weights
model.load_state_dict(torch.load('checkpoints/falcon_net.pth'))

# Set to evaluation mode
model.eval()

# Process image
with torch.no_grad():
    prediction = model(image)
    is_fake = prediction.argmax().item() == 1
```

## 📊 Performance

![Generalization Performance](assets/generalization_univfd_ap.png)

*Generalization performance comparison showing FALCON-Net's ability to detect images from unseen generative models.*

FALCON-Net achieves state-of-the-art performance in detecting AI-generated images across various datasets and generative models, demonstrating:

- High detection accuracy on known generative models
- Strong generalization to unseen generative models
- Robustness against post-processing operations
- Efficient inference speed

## 📁 Project Structure

```
FALCON-Net/
├── README.md
├── LICENSE
├── assets/
│   ├── overview.png          # Model architecture diagram
│   └── generalization_univfd_ap.png
├── src/
│   ├── models/              # Model implementations
│   ├── modules/             # INP and LVP modules
│   ├── utils/               # Utility functions
│   └── train.py             # Training script
├── checkpoints/             # Pre-trained models
├── data/                    # Dataset configurations
├── requirements.txt          # Python dependencies
└── configs/                 # Configuration files
```

## 🔬 Research Details

### Technical Contributions

1. **Novel Feature Extraction**: Introduces complementary INP and LVP modules for comprehensive forensic analysis
2. **Frequency Domain Analysis**: Leverages FFT-based noise pattern isolation for sensor fingerprint detection
3. **Local Pixel Modeling**: Captures unnatural regularities through directional pixel intensity analysis
4. **Robust Architecture**: Pruned ResNet backbone ensures efficient and effective feature processing

### Applications

- **Content Authenticity Verification**: Detect AI-generated images in social media and news
- **Forensic Analysis**: Investigate image origins and manipulation
- **Quality Assessment**: Evaluate generative model outputs
- **Security Systems**: Integrate into content moderation pipelines

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{falcon_net_2024,
  title={FALCON-Net: Feature Aggregation for Localized Context and Noise Network for AI-Generated Image Detection},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions to improve FALCON-Net! Please feel free to:

- Submit issues and feature requests
- Contribute code improvements
- Share experimental results
- Suggest architectural enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the research community for their valuable feedback and the open-source community for providing excellent tools and frameworks that made this research possible.

## 📞 Contact

For questions, suggestions, or collaborations, please contact:

- **Email**: your.email@institution.edu
- **Project Page**: [GitHub Repository](https://github.com/your-username/FALCON-Net)
- **Paper**: [arXiv/Conference Link]

---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.

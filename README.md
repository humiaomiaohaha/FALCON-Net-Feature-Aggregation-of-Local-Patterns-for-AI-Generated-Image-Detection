# FALCON-Net-Feature-Aggregation-of-Local-Patterns-for-AI-Generated-Image-Detection



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
git clone https://github.com/humiaomiaohaha/FALCON-Net-Feature-Aggregation-of-Local-Patterns-for-AI-Generated-Image-Detection.git
cd FALCON-Net

# Install dependencies
pip install -r requirements.txt
```




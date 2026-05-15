# EMG Gesture Recognition using Random Forest and SVM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.1%25-brightgreen.svg)]()

## 📌 Overview

This project implements a complete machine learning pipeline for recognizing hand gestures from surface electromyography (EMG) signals. The system classifies 7 different hand gestures using data collected from the Myo Thalmic bracelet, which captures muscle activity through 8 sensors placed around the forearm.

**Key Results:**
- ✅ Random Forest Accuracy: **96.1%**
- ✅ SVM Accuracy: **93.4%**
- ✅ Real-time capable (<100ms inference)

## 🎯 Problem Statement

Controlling robotic prostheses and assistive devices intuitively remains a challenge in human-robot interaction. This project addresses the problem by building classifiers that translate raw EMG signals into gesture commands, enabling natural control of robotic hands without manual switches or voice commands.

## 🦾 Robotics Applications

| Application | Description |
|-------------|-------------|
| **Prosthetic Hand Control** | Intuitive control for upper-limb amputees |
| **Human-Robot Interaction** | Gesture-based commands for collaborative robots |
| **Rehabilitation Robotics** | Monitor patient progress during hand therapy |
| **Teleoperation** | Remote control of robotic arms in hazardous environments |

## 📊 Dataset

**Source:** [UCI Machine Learning Repository - EMG Data for Gestures](https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures)

**Hardware:** Myo Thalmic Bracelet
- 8 EMG sensors equally spaced around the forearm
- Bluetooth connection
- 200 Hz sampling rate

**Statistics:**
| Attribute | Value |
|-----------|-------|
| Subjects | 36 |
| Gestures | 7 |
| Gesture duration | 3 seconds ON / 3 seconds OFF |
| Repetitions | 2 per subject |
| Total samples | 500,000 |

### The 7 Gestures

| Class | Gesture |
|-------|---------|
| 1 | Hand at rest |
| 2 | Hand clenched in a fist |
| 3 | Wrist flexion |
| 4 | Wrist extension |
| 5 | Radial deviation |
| 6 | Ulnar deviation |
| 7 | Extended palm |

*Note: Class 0 = Unlabeled data (excluded from training)*

## 🔧 Methodology

### Pipeline Overview

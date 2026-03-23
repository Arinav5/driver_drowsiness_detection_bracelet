# Driver Drowsiness Detection Wearable System

## Project Description

This project is a real-time driver monitoring system that detects signs of drowsiness and unsafe driving behavior. It uses computer vision to analyze facial features and driving patterns, then triggers a wearable device that provides haptic feedback through vibration.

The system focuses on two main safety signals:

* Driver drowsiness detection using facial landmarks
* Lane tracking to detect when a vehicle drifts out of its lane

When either condition is detected, a wearable wrist device vibrates to alert the driver.

The goal is to create a low-cost, responsive safety system that combines software and embedded hardware.

---

## Table of Contents

* Project Description
* System Architecture
* Hardware Components
* Software Components
* Installation
* Running the Project
* Usage

---

## System Architecture

The system is divided into two main parts:

1. Vision Processing Unit
   Runs on a Raspberry Pi and processes camera input using MediaPipe.

2. Wearable Feedback Device
   A wrist-mounted device built with a microcontroller and vibration motor.

### Data Flow

1. Camera captures driver video
2. MediaPipe processes facial landmarks
3. Drowsiness and lane deviation are detected
4. Raspberry Pi sends signal via Bluetooth or WiFi
5. Microcontroller activates vibration motor

---

## Hardware Components

* Raspberry Pi
* Camera module or USB webcam
* ESP32 microcontroller (compact version such as XIAO ESP32C3)
* Coin vibration motor
* 3.7V LiPo battery
* TP4056 charging module
* Nylon Velcro wrist strap
* Wires and basic electronic components (transistor, resistor)

---

## Software Components

* Python 3
* MediaPipe
* OpenCV
* MicroPython or Arduino framework for ESP32
* Bluetooth or WiFi communication library

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/driver-drowsiness-wearable.git
cd driver-drowsiness-wearable
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 3. Set up Raspberry Pi camera

Enable the camera interface:

```bash
sudo raspi-config
```

Then enable Camera and reboot.

---

## Running the Project

### 1. Start the detection system

```bash
python main.py
```

This script will:

* Capture video input
* Detect facial landmarks
* Monitor eye closure and head position
* Track lane boundaries

### 2. Connect to wearable device

Ensure the ESP32 is powered and running the firmware.

The Raspberry Pi will send a signal when:

* Eyes remain closed for a threshold duration
* Head tilts indicating drowsiness
* Vehicle drifts outside lane boundaries

---

## Usage

1. Wear the wrist device securely using the Velcro strap
2. Position the camera to clearly capture the driver's face and road view
3. Start the system on the Raspberry Pi
4. Begin driving simulation or testing

### System behavior

* If the driver shows signs of drowsiness, the wrist device vibrates
* If the vehicle leaves its lane, the wrist device vibrates
* Alerts are immediate and repeat until normal conditions resume

---

## Notes

* The system is designed for prototyping and educational use
* Lighting conditions and camera placement affect accuracy
* Battery size determines wearable runtime
* Lane tracking performance depends on camera angle and environment

---


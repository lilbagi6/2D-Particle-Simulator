# 2D Particle Simulator (Newton's Physics)
 
**This is the initial version of the simulation — I will update it soon**

---

## 📌 Project Overview

This project is a basic, yet interesting **2D simulation of particle dynamics**, built using Python and the `pygame` library.  
It models interactions between particles of different types (A, B, C), each with unique physical properties such as:

- Mass  
- Radius  
- Charge (positive, negative, or neutral)  
- Initial velocity  
- Lennard-Jones potential parameters  

The simulation features realistic physical forces and visualizes particle movement and interaction in real-time.

## Simulation Use Cases
This simulator can be used to:
- Study particle clustering and pattern formation
- Visualize how forces influence multi-particle systems
- Test Lennard-Jones and electrostatic models
- Explore emergent behavior in dynamic environments
- Teach or demonstrate basic concepts in molecular dynamics

---

##  Simulated Forces

| Force                 | Toggle Key | Description                                                                 |
|----------------------|------------|-----------------------------------------------------------------------------|
| **Lennard-Jones**    | Always on  | Short-range interaction: attraction and repulsion between particles.        |
| **Coulomb Force**    | Always on  | Electrostatic force between charged particles (+/− attract, ± repel).       |
| **Electric Field**   | `E`        | Applies a constant force along the X-axis, affecting only charged particles.|
| **Lorentz Force**    | `L`        | Applies force based on velocity × magnetic field (`v × B`).                |

Particles experience acceleration and direction changes based on the sum of these forces.

---

##  Controls

| Key | Action                              |
|-----|-------------------------------------|
| `R` | Reset and regenerate all particles  |
| `T` | Toggle motion trails on/off         |
| `E` | Toggle electric field on/off        |
| `L` | Toggle Lorentz force on/off         |

---

## ⚙️ Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/libagi6/2D-Particle-Simulator.git
cd 2D-Particle-Simulator
pip install -r requirements.txt

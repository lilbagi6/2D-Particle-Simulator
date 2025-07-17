import pygame
import numpy as np

# ----------------- Window & World Params -----------------
WINDOW_SIZE = 800             # Window size in pixels
WORLD_SIZE = 10.0             # World size in nanometers (nm)
SCALE = WINDOW_SIZE / WORLD_SIZE  # Scale factor for converting world coordinates to screen pixels
BACKGROUND_COLOR = (0, 0, 0)  # Black

# ----------------- Time steps -----------------
DT = 0.005                    # Time step for simulation (in seconds)
FPS = 60                      # Frames per second for the simulation

# ----------------- Particle A -----------------
NUM_PARTICLES_A = 10
PARTICLE_MASS_A = 1e-27         # Mass in kg
PARTICLE_RADIUS_A = 0.07        # Radius in nanometers
PARTICLE_COLOR_A = (255, 0, 0)  # Red (RGB)
INITIAL_SPEED_MAX_A = 0.5       # Maximum initial speed in nm/s

# ----------------- Particle B -----------------
NUM_PARTICLES_B = 10
PARTICLE_MASS_B = 5e-27
PARTICLE_RADIUS_B = 0.08
PARTICLE_COLOR_B = (0, 255, 0)
INITIAL_SPEED_MAX_B = 0.4 

# ----------------- Particle C -----------------
NUM_PARTICLES_C = 10
PARTICLE_MASS_C = 0.5e-27
PARTICLE_RADIUS_C = 0.05
PARTICLE_COLOR_C = (0, 120, 255)         
INITIAL_SPEED_MAX_C = 0.7  

# ----------------- Physic constants -----------------
EPSILON_0 = 8.854e-12         # Electric constant (F/m)
E_CHARGE = 1.602e-19          # Elementary charge (C)
G_const = 6.67430e-11         # Gravitational constant (m³·kg⁻¹·s⁻²)
K_E = 8.9875e9                # Coulomb's constant (N·m²/C²)
B_vec = np.array([0, 0, 1e-10])     # Magnetic field in z-direction (For Lorentz force)
E_FIELD = np.array([1e-11, 0])      # Weak Electric field to the right in the X-axis -->

# ----------------- Lennard-Jones potential params -----------------
LJ_EPSILON_A = 1e-27              # Energy well depth (in J)
LJ_SIGMA_A = 0.2                  # Distance at which the potential is zero (in nm)
LJ_CUTOFF_A = 2.5 * LJ_SIGMA_A    # Cutoff distance for interaction (in nm)

LJ_EPSILON_B = 0.5e-27
LJ_SIGMA_B = 0.25
LJ_CUTOFF_B = 2.5 * LJ_SIGMA_B

LJ_EPSILON_C = 1.2e-27
LJ_SIGMA_C = 0.15
LJ_CUTOFF_C = 2.5 * LJ_SIGMA_C

# ----------------- Other Params -----------------
DAMPING = 0.95
simulation_time = 0.0
show_trails = False
enable_e_field = False
enable_lorentz = True

# -----------------Define the particle class -----------------
class Particle:
    def __init__(self, position, velocity, mass, radius, color, charge=0.0, p_type: str = 'A'):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2, dtype= float)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.charge = charge
        self.type = p_type

        # Lennard-Jones parameters based on particle type
        if p_type == 'A':
            self.lj_sigma = LJ_SIGMA_A
            self.lj_epsilon = LJ_EPSILON_A
            self.lj_cutoff = LJ_CUTOFF_A
        elif p_type == 'B':
            self.lj_sigma = LJ_SIGMA_B
            self.lj_epsilon = LJ_EPSILON_B
            self.lj_cutoff = LJ_CUTOFF_B
        elif p_type == 'C':
            self.lj_sigma = LJ_SIGMA_C
            self.lj_epsilon = LJ_EPSILON_C
            self.lj_cutoff = LJ_CUTOFF_C

        self.trail = [] 
        self.max_trail_length = 100

    # Update position and velocity using Verlet integration
    def verlet_position(self, dt):
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += 0.5 * self.acceleration * dt
        # Trail management
        self.trail.append(self.position.copy())
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

    def verlet_velocity(self, dt):
        self.velocity += 0.5 * self.acceleration * dt
        #self.velocity *= DAMPING

    # Border collision detection
    def handle_border_collision(self):
        for i in range(2):  # 0:x and 1:y
            if self.position[i] - self.radius < 0:
                self.position[i] = self.radius
                self.velocity[i] *= -1 # Or use -DAMPING to simulate energy loss

            elif self.position[i] + self.radius > WORLD_SIZE:
                self.position[i] = WORLD_SIZE - self.radius
                self.velocity[i] *= -1 # Or use -DAMPING to simulate energy loss

    def draw(self, screen):
        x, y = self.position * SCALE
        pygame.draw.circle(screen, self.color, (int(x), int(y)), int(self.radius * SCALE))
        # Draw trail if enabled
        if show_trails:
            for i in range(1, len(self.trail)):
                pygame.draw.line(screen, self.color,
                                self.trail[i-1] * SCALE,
                                self.trail[i] * SCALE, 1)
            
#  -----------------Function to create a random particles -----------------
def generate_particles():
    particles = []

    def create_random_particle(p_type, mass, radius, color, speed_max):
        position = np.random.uniform(radius, WORLD_SIZE - radius, size=2)
        velocity = np.random.uniform(-speed_max, speed_max, size=2)

        if p_type == 'A':
            charge = +E_CHARGE
        elif p_type == 'B':
            charge = np.random.choice([-E_CHARGE, 0.0, +E_CHARGE])
        elif p_type == 'C':
            charge = -E_CHARGE

        return Particle(position, velocity, mass, radius, color, charge, p_type)

    for _ in range(NUM_PARTICLES_A):
        particles.append(create_random_particle('A', PARTICLE_MASS_A, PARTICLE_RADIUS_A, PARTICLE_COLOR_A, INITIAL_SPEED_MAX_A))

    for _ in range(NUM_PARTICLES_B):
        particles.append(create_random_particle('B', PARTICLE_MASS_B, PARTICLE_RADIUS_B, PARTICLE_COLOR_B, INITIAL_SPEED_MAX_B))

    for _ in range(NUM_PARTICLES_C):
        particles.append(create_random_particle('C', PARTICLE_MASS_C, PARTICLE_RADIUS_C, PARTICLE_COLOR_C, INITIAL_SPEED_MAX_C))

    return particles

# ----------------- Coulomb's force -----------------
def apply_coulomb_force(p1, p2):
    q1 = p1.charge
    q2 = p2.charge
    if q1 == 0 or q2 == 0:
        return 
    
    r_vec = p1.position - p2.position
    r = np.linalg.norm(r_vec)
    if r == 0:
        return
    
    force_mag = K_E * (q1 * q2) / (r**2)
    force_vec = force_mag * (r_vec / r)

    p1.acceleration += force_vec / p1.mass
    p2.acceleration -= force_vec / p2.mass
    
# ----------------- Gravitational forces -----------------
def apply_gravity(p1, p2):
    r_vec = p1.position - p2.position
    r = np.linalg.norm(r_vec)
    if r == 0:
        return
    
    force_mag = G_const * (p1.mass * p2.mass) / (r**2)
    force_vec = force_mag * (r_vec / r) # Normalize the vector

    p1.acceleration += force_vec / p1.mass
    p2.acceleration -= force_vec / p2.mass

# ----------------- Lorentz Force -----------------
def apply_lorentz_force(particle, B_vec): # Magnetic field in z-direction
    if particle.charge == 0:
        return
    
    v3d = np.array([*particle.velocity, 0]) # Convert to 3D vector
    force_3d = particle.charge * np.cross(v3d, B_vec)  # Lorentz force F = q(v x B)
    particle.acceleration += force_3d[:2] / particle.mass # Only take x and y components    

# ----------------- Compute forces function -----------------
def compute_forces(particles):
    for p in particles:
        p.acceleration[:] = 0  

    # Calculate forces between particles
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            p1 = particles[i]
            p2 = particles[j]
            r_vec = p1.position - p2.position
            r = np.linalg.norm(r_vec)

            # Lennard-Jones params
            sigma_ij = 0.5 * (p1.lj_sigma + p2.lj_sigma)
            epsilon_ij = np.sqrt(p1.lj_epsilon * p2.lj_epsilon)
            cutoff_ij = 2.5 * sigma_ij

            # Calculate Lennard-Jones force
            if r < cutoff_ij and r > 1e-12:
                sr6 = (sigma_ij / r) ** 6
                sr12 = sr6 ** 2
                force_mag = 24 * epsilon_ij * (2 * sr12 - sr6) / r**2
                force = force_mag * r_vec 
            else:
                force = np.zeros(2)

            # Normalize force to avoid numerical instability
            force_magnitude = np.linalg.norm(force)
            max_force = 1e-20 # Limit the force to avoid numerical instability
            if force_magnitude > max_force:
                force *= max_force / (force_magnitude + 1e-10)

            # Apply forces to both particles: F = ma => a = F/m
            p1.acceleration += force / p1.mass
            p2.acceleration -= force / p2.mass

            # Apply Coulomb's force if particles have charge
            apply_coulomb_force(p1, p2)
            # Apply gravitational force
            apply_gravity(p1, p2)
            # Apply Lorentz force
            if enable_lorentz:
                apply_lorentz_force(p1, B_vec)
            # Apply electric field force
            if enable_e_field:
                for p in particles:
                    if p.charge != 0:
                        force_e = p.charge * E_FIELD # F = qE
                        p.acceleration += force_e / p.mass

# -----------------Main simulation loop -----------------
pygame.init()
font = pygame.font.SysFont("Arial", 12)
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()
start_ticks = pygame.time.get_ticks()

def draw_text(surface, text, pos, color=(150, 150, 150)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos) 

particles = generate_particles()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                show_trails = not show_trails
            elif event.key == pygame.K_r:
                particles = generate_particles()
                simulation_time = 0.0
            elif event.key == pygame.K_e:
                enable_e_field = not enable_e_field
            elif event.key == pygame.K_l:
                enable_lorentz = not enable_lorentz
        
    screen.fill(BACKGROUND_COLOR)
    real_time = (pygame.time.get_ticks() - start_ticks) / 1000.0

    draw_text(screen, "R - Reset Particles", (10, 10))
    draw_text(screen, "T - Trails: " + ("ON" if show_trails else "OFF"), (10, 30))
    draw_text(screen, "E - Electric Field: " + ("ON" if enable_e_field else "OFF"), (10, 50))
    draw_text(screen, "L - Lorentz Force: " + ("ON" if enable_lorentz else "OFF"), (10, 70))
    draw_text(screen, f"Simulation Time (DT): {simulation_time:.2f} s", (10, WINDOW_SIZE - 30))
    draw_text(screen, f"Real Time: {real_time:.2f} s", (10, WINDOW_SIZE - 50))

    for p in particles:
        p.verlet_position(DT)

    compute_forces(particles)
    simulation_time += DT 

    for p in particles:
        p.verlet_velocity(DT)
        p.handle_border_collision()
        p.draw(screen)

    pygame.display.flip()
    clock.tick(FPS) 

pygame.quit() 
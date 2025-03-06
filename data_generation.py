import torch

def generate_data(x1range, x2range, numICs, mu, lam, T, dt, seed):
   # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random initial conditions for x1 and x2
    x1 = (x1range[1] - x1range[0]) * torch.rand(numICs) + x1range[0]
    x2 = (x2range[1] - x2range[0]) * torch.rand(numICs) + x2range[0]
    u = torch.rand(numICs, T) - 0.5

    dt_lam = dt * lam

    # Preallocate xu with shape [numICs, lenT, 3]
    xuk = torch.zeros(numICs, T, 3, dtype=torch.float32)

    xuk[:, :, 2] = u

    for t in range(T):

        xuk[:, t, 0] = x1
        xuk[:, t, 1] = x2

        dx1 = dt * mu * x1 + dt*u[:, t-1]
        dx2 = dt_lam * (x2 - x1**2)

        x1 += dx1
        x2 += dx2

    return xuk



def DataGenerator(x1range, x2range, numICs, mu, lam, T, dt):

    # Create test, validation, and training tensors with different percentages of numICs
    seed = 1
    test_tensor = generate_data(x1range, x2range, round(0.1 * numICs), mu, lam, T, dt, seed)

    seed = 2
    val_tensor = generate_data(x1range, x2range, round(0.2 * numICs), mu, lam, T, dt, seed)

    seed = 3
    train_tensor = generate_data(x1range, x2range, round(0.7 * numICs), mu, lam, T, dt, seed)

    return train_tensor, test_tensor, val_tensor

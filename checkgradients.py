def check_gradients(gradients, name):
    for param_name, grad in gradients.items():
        grad_norm = np.linalg.norm(grad)
        grad_max = np.max(np.abs(grad))
        print(f"{name} - {param_name}: norm={grad_norm:.6f},
        max={grad_max:.6f}")

        if grad_norm > 10:
            print(f"  WARNING: Large gradient norm in {param_name}")
        if grad_max < 1e-6:
            print(f"  WARNING: Very small gradients in {param_name}")
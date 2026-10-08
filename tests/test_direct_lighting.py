import numpy as np


def compute_direct_lighting_numpy(
    surface_point: np.ndarray,
    normal: np.ndarray,
    albedo: np.ndarray,
    light_pos: np.ndarray,
    light_intensity: np.ndarray,
    ambient_color: np.ndarray,
    is_occluded: bool = False,
    use_attenuation: bool = True,
    min_dist: float = 0.3,
) -> np.ndarray:
    """NumPy reference implementation of Lambertian direct lighting with shadow rays."""
    normal_unit = normal / np.linalg.norm(normal)
    to_light = light_pos - surface_point
    dist = float(np.linalg.norm(to_light))
    light_dir = to_light / dist

    # Ambient term
    illumination = albedo * ambient_color

    if not is_occluded:
        cos_theta = max(0.0, float(np.dot(normal_unit, light_dir)))
        if cos_theta > 0.0:
            atten = 1.0 / (max(dist, min_dist) ** 2) if use_attenuation else 1.0
            illumination += cos_theta * atten * (albedo * light_intensity)

    return illumination


def test_direct_lighting_properties_numpy():
    """Verify Lambert direct lighting follows cosine law and occlusion."""
    pt = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    albedo = np.array([1.0, 1.0, 1.0])
    ambient = np.array([0.1, 0.1, 0.1])
    intensity = np.array([1.0, 1.0, 1.0])

    # 1. Directly overhead unoccluded
    light_overhead = np.array([0.0, 5.0, 0.0])
    l_overhead = compute_direct_lighting_numpy(
        pt,
        normal,
        albedo,
        light_overhead,
        intensity,
        ambient,
        is_occluded=False,
        use_attenuation=False,
    )
    # Cos(0) = 1.0 => 0.1 + 1.0 = 1.1
    assert np.allclose(l_overhead, 1.1, atol=1e-12)

    # 2. Directly overhead occluded (shadow)
    l_shadow = compute_direct_lighting_numpy(
        pt,
        normal,
        albedo,
        light_overhead,
        intensity,
        ambient,
        is_occluded=True,
        use_attenuation=False,
    )
    # Only ambient
    assert np.allclose(l_shadow, 0.1, atol=1e-12)

    # 3. Horizon light (cos = 0)
    light_horizon = np.array([5.0, 0.0, 0.0])
    l_horizon = compute_direct_lighting_numpy(
        pt,
        normal,
        albedo,
        light_horizon,
        intensity,
        ambient,
        is_occluded=False,
        use_attenuation=False,
    )
    assert np.allclose(l_horizon, 0.1, atol=1e-12)


def test_distance_attenuation_numpy():
    """Verify inverse square distance attenuation."""
    pt = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    albedo = np.array([1.0, 1.0, 1.0])
    ambient = np.array([0.0, 0.0, 0.0])
    intensity = np.array([4.0, 4.0, 4.0])

    # Light at distance 2
    l_dist2 = compute_direct_lighting_numpy(
        pt,
        normal,
        albedo,
        np.array([0.0, 2.0, 0.0]),
        intensity,
        ambient,
        is_occluded=False,
        use_attenuation=True,
    )
    # atten = 1 / 4, intensity = 4 => cos(0) * (1/4) * 4 = 1.0
    assert np.allclose(l_dist2, 1.0, atol=1e-12)

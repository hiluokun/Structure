import numpy as np
from ase import Atoms
from ase.io import write

# --------------------
# 基础参数设定
# --------------------
a1_l = np.array([0.5, 0.8660254, 0.0])
a2_l = np.array([-0.5, 0.8660254, 0.0])
alat = 2.43
a1_ang = a1_l * alat
a2_ang = a2_l * alat

# 修改后的层间距
t_z = 3.4

# 修改后的晶胞高度（z 方向）
cell_height = 15.0

# --------------------
# 构建 supercell 的函数
# --------------------
def generate_supercell(size=30, z=0.0): ## 当结果出现空白地方的时候说明需要更大的size
    # 基元原子位置
    basis = [np.array([0, 0, 0]), (a1_ang + a2_ang) / 3]
    atoms = []
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            for b in basis:
                pos = i * a1_ang + j * a2_ang + b
                atoms.append([pos[0], pos[1], z])
    return np.array(atoms)

def rotate_around_center(atoms, theta):
    center = np.mean(atoms[:, :2], axis=0)
    atoms_shifted = atoms.copy()
    atoms_shifted[:, :2] -= center
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    atoms_rotated = atoms_shifted.copy()
    atoms_rotated[:, :2] = atoms_shifted[:, :2] @ R.T
    atoms_rotated[:, :2] += center
    return atoms_rotated

def filter_atoms(atoms, T1, T2):
    inv = np.linalg.inv(np.array([T1[:2], T2[:2]]).T)
    selected = []
    for pos in atoms:
        frac = inv @ pos[:2]
        if np.all((frac >= -1e-5) & (frac < 1 - 1e-5)):
            selected.append(pos)
    return selected

# --------------------
# 控制选项设置（角度范围以°为单位）
# --------------------
max_atoms = 200000   # 理论上 Moiré 单元内原子数上限
min_angle_deg = 0.2    # 角度下限（°）
max_angle_deg = 0.3    # 角度上限（°）

# 内部转换为弧度
min_angle = np.radians(min_angle_deg)
max_angle = np.radians(max_angle_deg)

# 增加枚举 m, n 的上限（注意：较小转角通常需要较大的 m,n 值）
max_m = 500   # 可根据需要进一步调整

# --------------------
# 枚举可能的 SL1 参数
# 这里为避免重复，仅考虑 m <= n 的情况
possible_SL1 = []
for m in range(1, max_m+1):
    for n in range(m, max_m+1):
        atom_count = 4 * (m**2 + m*n + n**2)
        if atom_count <= max_atoms:
            # 转角计算公式
            #   cos(theta) = (m^2 + 4*m*n + n^2) / [2*(m^2 + m*n + n^2)]
            cos_theta = (m**2 + 4 * m * n + n**2) / (2 * (m**2 + m*n + n**2))
            cos_theta = min(cos_theta, 1.0)  # 数值修正
            angle_rad = np.arccos(cos_theta)
            if min_angle <= angle_rad <= max_angle:
                possible_SL1.append((m, n, atom_count, angle_rad))

# 输出符合条件的 SL1 参数信息（显示转角为°）
print("符合条件（原子数 ≤ {}）且转角在 {}° 到 {}° 的 SL1 选项:".format(max_atoms, min_angle_deg, max_angle_deg))
for m, n, atom_count, angle in possible_SL1:
    print(f"SL1 = ({m},{n}), 预计原子数: {atom_count}, 转角: {np.degrees(angle):.4f}°")

# --------------------
# 设置 z 方向居中及生成结构并保存
# --------------------
z_center = cell_height / 2.0  # 晶胞 z 方向中心

for m, n, atom_count, angle_rad in possible_SL1:
    SL1 = (m, n)
    # 根据构造规则，SL2 定义为 (-n, m+n)
    SL2 = (-n, m+n)
    
    # 构造 Moiré 单元矢量及晶胞
    T1 = m * a1_ang + n * a2_ang
    T2 = SL2[0] * a1_ang + SL2[1] * a2_ang
    cell = np.array([
        [T1[0], T1[1], 0],
        [T2[0], T2[1], 0],
        [0,      0,    cell_height]
    ])
    
    # 生成结构，将两层在 z 方向居中放置
    layer1 = generate_supercell(z=z_center - t_z/2)
    layer2 = generate_supercell(z=z_center + t_z/2)
    layer2_rot = rotate_around_center(layer2, angle_rad)
    
    layer1_cut = filter_atoms(layer1, T1, T2)
    layer2_cut = filter_atoms(layer2_rot, T1, T2)
    positions = layer1_cut + layer2_cut
    symbols = ['C'] * len(positions)
    
    atoms_obj = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    fname = f"TBG_SL1_{m}_{n}_SL2_{SL2[0]}_{SL2[1]}_{np.degrees(angle_rad):.4f}deg_{atom_count}atoms.xyz"
    write(fname, atoms_obj)
    print(f"✓ 已生成 {fname}: SL1=({m},{n}), 转角 = {np.degrees(angle_rad):.4f}°, 实际原子数 = {len(positions)}")

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
def generate_supercell(size=30, z=0.0):
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
# 控制选项设置
# --------------------
max_atoms = 120         # 理论上 Moiré 单元内原子数上限（公式：4*(m^2+m*n+n^2)）
min_angle = 0.0         # 角度下限（弧度）
max_angle = np.pi / 2   # 角度上限（弧度）

# --------------------
# 枚举可能的 SL1 参数
# 这里为避免重复，仅考虑 m <= n 的情况
possible_SL1 = []
for m in range(1, 10):      # m 的取值范围可调整
    for n in range(m, 10):  # n 从 m 到上限
        atom_count = 4 * (m**2 + m * n + n**2)
        if atom_count <= max_atoms:
            cos_theta = (m**2 + 4 * m * n + n**2) / (2 * (m**2 + m * n + n**2))
            # 数值误差修正
            cos_theta = min(cos_theta, 1.0)
            angle_rad = np.arccos(cos_theta)
            if min_angle <= angle_rad <= max_angle:
                possible_SL1.append((m, n, atom_count, angle_rad))

# 输出符合条件的 SL1 参数信息
print("符合条件（原子数 ≤ {}）的 SL1 选项:".format(max_atoms))
for m, n, atom_count, angle in possible_SL1:
    print(f"SL1 = ({m},{n}), 预计原子数: {atom_count}, 转角: {np.degrees(angle):.2f}°")

# --------------------
# 设置 z 方向居中及生成结构并保存
# --------------------
# 计算晶胞 z 方向中心位置
z_center = cell_height / 2.0  # 7.5 Å

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
    
    # 生成结构：层1 与层2 分别放置在晶胞中心的下/上半部分
    layer1 = generate_supercell(z=z_center - t_z/2)  # 约 7.5 - 1.7 = 5.8 Å
    layer2 = generate_supercell(z=z_center + t_z/2)  # 约 7.5 + 1.7 = 9.2 Å
    # 旋转第二层
    layer2_rot = rotate_around_center(layer2, angle_rad)
    
    # 截取 Moiré 单元内的原子
    layer1_cut = filter_atoms(layer1, T1, T2)
    layer2_cut = filter_atoms(layer2_rot, T1, T2)
    positions = layer1_cut + layer2_cut
    symbols = ['C'] * len(positions)
    
    atoms_obj = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    fname = f"TBG_SL1_{m}_{n}_SL2_{SL2[0]}_{SL2[1]}_{np.degrees(angle_rad):.2f}deg_{atom_count}atoms.xyz"
    write(fname, atoms_obj)
    print(f"✓ 已生成 {fname}: SL1=({m},{n}), 转角 = {np.degrees(angle_rad):.2f}°, 实际原子数 = {len(positions)}")

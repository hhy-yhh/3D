import pandas as pd

df = pd.read_csv('/data/huanghaoyang/3D/database/metadata.csv')

def generate_caption(row):
    return (
        f"An automotive brake caliper: the fixing center distance (tangential_fixing_interaxis, corresponding to fixing side / fixing hole features) is {row['tangential_fixing_interaxis']:.2f} mm; "
        f"the inner pad tangential position (tangential_pad_inner, corresponding to fixing side features) is {row['tangential_pad_inner']:.2f} mm; "
        f"the outer pad tangential position (tangential_pad_outer, corresponding to outer side features) is {row['tangential_pad_outer']:.2f} mm; "
        f"the disc inner radius (radial_disc_internal_radius, corresponding to tie rod / bridge inlet / bridge outlet features) is {row['radial_disc_internal_radius']:.2f} mm. "
        f"The number of pistons (pistons_number, corresponding to piston hole features) is {int(row['pistons_number'])}; "
        f"the piston inlet diameter (diameter_pistons_inlet, corresponding to piston hole features) is {row['diameter_pistons_inlet']:.2f} mm; "
        f"the piston central diameter (diameter_pistons_central, corresponding to piston hole features) is {row['diameter_pistons_central']:.2f} mm; "
        f"the piston outlet diameter (diameter_pistons_outlet, corresponding to piston hole features) is {row['diameter_pistons_outlet']:.2f} mm; "
        f"the piston effective radius (pistons_effective_radius, corresponding to piston hole features) is {row['pistons_effective_radius']:.2f} mm. "
        f"The disc thickness (axial_disc_thickness, corresponding to tie rod / bridge inlet / bridge outlet features) is {row['axial_disc_thickness']:.2f} mm; "
        f"the external radius (radial_space_external_radius, corresponding to tie rod / bridge inlet / bridge outlet features) is {row['radial_space_external_radius']:.2f} mm; "
        f"the internal radius (radial_space_internal_radius, corresponding to tie rod / bridge inlet / bridge outlet features) is {row['radial_space_internal_radius']:.2f} mm; "
        f"the radial cut (radial_space_cut, corresponding to fixing side / outer side features) is {row['radial_space_cut']:.2f} mm. "
        f"The axial distance between the two fixing sides (axial_disc_distance, corresponding to fixing side features) is {row['axial_disc_distance']:.2f} mm; "
        f"the tangential space dimension (tangential_space_dimension, corresponding to fixing side / outer side features) is {row['tangential_space_dimension']:.2f} mm; "
        f"the axial space dimension (axial_space_dimension, corresponding to fixing side / outer side features) is {row['axial_space_dimension']:.2f} mm; "
        f"the radial space dimension (radial_space_dimension, corresponding to fixing side / outer side features) is {row['radial_space_dimension']:.2f} mm; "
        f"the total caliper volume (volume, corresponding to fixing side / outer side features) is {row['volume']:.6f} m³."
    )

df['captions'] = df.apply(generate_caption, axis=1)
df.to_csv('/data/huanghaoyang/3D/database/metadata.csv', index=False)
print('✅ 完成！已更新 metadata.csv')
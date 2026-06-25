import transformers
from pathlib import Path

HELPER_CODE = '''
def interpolate_rope_2d(cos, sin, target_h, target_w):
    """
    Interpolate 2D RoPE embeddings to match target spatial resolution.
    cos, sin: [seq_len, head_dim]
    """
    import torch.nn.functional as F

    seq_len, dim = cos.shape
    src = int(seq_len ** 0.5)

    if src * src != seq_len:
        return cos, sin

    cos = cos.view(src, src, dim)
    sin = sin.view(src, src, dim)

    cos = F.interpolate(
        cos.permute(2, 0, 1).unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)

    sin = F.interpolate(
        sin.permute(2, 0, 1).unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)

    return (
        cos.reshape(target_h * target_w, dim),
        sin.reshape(target_h * target_w, dim),
    )
'''

PATCH_MARKER = "def interpolate_rope_2d("


def find_modeling_file():
    # Let Python tell us exactly where the module is installed
    base_path = Path(transformers.__file__).parent
    path = base_path / "models" / "sam3" / "modeling_sam3.py"
    
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find modeling_sam3.py at {path}")


def main():
    path = find_modeling_file()
    print(f"Found modeling file at: {path}")
    text = path.read_text(encoding="utf-8")

    # Already patched?
    if PATCH_MARKER in text:
        print("✓ SAM-3 already patched!")
        return

    # Insert helper AFTER imports (first blank line after imports)
    lines = text.splitlines()
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            insert_idx = i + 1
            break

    lines.insert(insert_idx, HELPER_CODE.strip())
    text = "\n".join(lines)

    # Patch rotary application site
    old = "query, key = apply_rotary_pos_emb_2d(query, key, cos=cos, sin=sin)"
    new = (
        "cos, sin = interpolate_rope_2d(cos, sin, height, width)\n"
        "        query, key = apply_rotary_pos_emb_2d(query, key, cos=cos, sin=sin)"
    )

    if old not in text:
        raise RuntimeError("Could not find rotary application site in the file. It might be a different transformers version.")

    text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8")
    print("✓ Successfully patched SAM-3 interpolated RoPE!")


if __name__ == "__main__":
    main()
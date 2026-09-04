from pathlib import Path

import cv2
import numpy as np

from src.main import main


def test_cli_processes_image(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "nested" / "output.png"
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(input_path), image)

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()
    assert "Processed image written" in captured.out


def test_cli_reports_input_error(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--input",
            str(tmp_path / "missing.png"),
            "--output",
            str(tmp_path / "output.png"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: Could not read input image" in captured.err

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from webplanner.data.dicom import load_3dra_series

_ORIENTATION = [0, 1, 0, 0, 0, 1]
_DIRECTION = (0, 0, 1, 1, 0, 0, 0, 1, 0)


def _dataset(path, uid, values):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = uid
    ds.Rows = ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.broadcast_to(np.array(values, dtype="<u2")[:, None, None], (len(values), 2, 2)).tobytes()
    return ds


@pytest.mark.parametrize("positions", [(12, 10), (10,)])
def test_stack_preserves_patient_geometry(tmp_path, positions):
    uid = generate_uid()
    for i, x in enumerate(positions):
        path = tmp_path / f"{i}.dcm"
        ds = _dataset(path, uid, [x])
        ds.ImageOrientationPatient = _ORIENTATION
        ds.ImagePositionPatient = [x, 20, 30]
        ds.PixelSpacing = [0.7, 0.4]
        ds.SliceThickness = 2
        ds.save_as(str(path))
    volume = load_3dra_series(tmp_path, uid)
    assert volume.origin == (30, 20, 10)
    assert volume.direction == _DIRECTION
    assert volume.spacing == (2, 0.7, 0.4)
    np.testing.assert_array_equal(volume.array[:, 0, 0], sorted(positions))


def test_enhanced_frames_preserve_sorted_origin_and_shared_geometry(tmp_path):
    uid = generate_uid()
    path = tmp_path / "enhanced.dcm"
    ds = _dataset(path, uid, [12, 10])
    ds.NumberOfFrames = 2
    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = _ORIENTATION
    shared.PlaneOrientationSequence = [orientation]
    measures = Dataset()
    measures.PixelSpacing = [0.7, 0.4]
    shared.PixelMeasuresSequence = [measures]
    ds.SharedFunctionalGroupsSequence = [shared]
    frames = []
    for x in (12, 10):
        frame = Dataset()
        position = Dataset()
        position.ImagePositionPatient = [x, 20, 30]
        frame.PlanePositionSequence = [position]
        frames.append(frame)
    ds.PerFrameFunctionalGroupsSequence = frames
    ds.save_as(str(path))

    volume = load_3dra_series(tmp_path, uid)
    assert volume.origin == (30, 20, 10)
    assert volume.direction == _DIRECTION
    assert volume.spacing == (2, 0.7, 0.4)
    np.testing.assert_array_equal(volume.array[:, 0, 0], [10, 12])

__version__ = '0.6.0.dev20200212'
git_version = 'cd117551a95ecaf7ed142101633f7b5d9bf49140'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()

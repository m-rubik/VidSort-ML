from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='vidsortml',
    version='0.0.1',
    description='ML for facial recognition',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/m-rubik/VidSort-ML',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['cmake', 'dlib', 'face-recognition', 'opencv-python', 'numpy', 'scikit-learn', 'PyQt5', 'selenium', 'webdriver-manager', 'matplotlib', 'Pillow', 'requests']
)
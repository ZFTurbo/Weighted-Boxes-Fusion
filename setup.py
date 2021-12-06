try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name='ensemble_boxes',
    version='1.0.8',
    author='Roman Solovyev (ZFTurbo)',
    packages=['ensemble_boxes', ],
    url='https://github.com/ZFTurbo/Weighted-Boxes-Fusion',
    license='MIT License',
    description='Python implementation of several methods for ensembling boxes from object detection models: Non-maximum Suppression (NMS), Soft-NMS, Non-maximum weighted (NMW), Weighted boxes fusion (WBF)',
    long_description='Python implementation of several methods for ensembling boxes from object detection models: Non-maximum Suppression (NMS), Soft-NMS, Non-maximum weighted (NMW), Weighted boxes fusion (WBF)'
                     'More details: https://github.com/ZFTurbo/Weighted-Boxes-Fusion',
    install_requires=[
        "numpy",
        "pandas",
        "numba",
    ],
)

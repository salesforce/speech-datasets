from setuptools import find_packages, setup

setup(
    name="speech_datasets",
    version="0.1.0",
    author="Aadyot Bhatnagar",
    author_email="abhatnagar@salesforce.com",
    license="Apache-2.0",
    packages=find_packages(include=["speech_datasets*"]),
    install_requires=[
        "h5py>=2.9.0",
        "humanfriendly",
        "Kaldiio",
        "numpy",
        "pillow>=6.1.0",
        "PyYAML>=5.1.2",
        "ray",
        "resampy",
        "scipy",
        "sentencepiece<0.1.90,>=0.1.82",
        "soundfile>=0.10.2",
        "tqdm",
        "typeguard>=2.7.0",
    ]
)

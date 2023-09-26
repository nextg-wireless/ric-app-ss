# ==================================================================================
#       Copyright (c) 2020 China Mobile Technology (USA) Inc. Intellectual Property.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ==================================================================================
from setuptools import setup, find_packages

setup(
    name="ss",
    version="0.3.0",
    packages=find_packages(exclude=["tests.*", "tests"]),
    author="O-RAN-SC Community + NextG Wireless Lab",
    description="Spectrum Sensing xApp for SenseORAN",
    install_requires=["ricxappframe==3.0.1", "p5py", "PEP517", "Cython", "numpy==1.24.3", "pandas==1.5.2", "torch==1.11.0", "torchvision>=0.10.0", "torchaudio>=0.9.0", "influxdb==5.3.1", "schedule==1.1.0", "pysctp"],
    entry_points={"console_scripts": ["start-ss.py=ss.main:start"]},  
    license="Apache 2.0 + GPL v3.0",
    data_files=[("", ["LICENSE.txt"])],
)

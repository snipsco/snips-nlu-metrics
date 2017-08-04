from __future__ import unicode_literals

import io
import os
from subprocess import check_output, CalledProcessError

import pip

PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with io.open(os.path.join(PACKAGE_PATH, "__default_training_version__")) as f:
    DEFAULT_TRAINING_VERSION = f.readline().strip()

with io.open(os.path.join(PACKAGE_PATH, "__default_inference_version__")) as f:
    DEFAULT_INFERENCE_VERSION = f.readline().strip()


def update_nlu_packages(snips_nlu_version, snips_nlu_rust_version):
    if snips_nlu_version is not None:
        update_package(package_name='snips_nlu', version=snips_nlu_version)
    if snips_nlu_rust_version is not None:
        update_package(package_name='snips_nlu_rust',
                       version=snips_nlu_rust_version)


def update_package(package_name, version):
    installed_packages = pip.get_installed_distributions()
    if any(package.key == package_name and package.version == version
           for package in installed_packages):
        return
    if any(package.key == package_name for package in installed_packages):
        uninstall_package(package_name)
    install_package(package_name, version)


def uninstall_package(package_name):
    try:
        check_output(['pip', 'uninstall', package_name, '-y'])
    except CalledProcessError as e:
        raise Exception("Error while uninstalling '%s': %s"
                        % (package_name, e.output))


def install_package(package_name, version):
    try:
        check_output(['pip', 'install', package_name + "==" + version])
    except CalledProcessError as e:
        raise Exception('Error while installing %s==%s package: %s'
                        % (package_name, version, e.output))


def get_last_repository_tag(repository_url):
    return check_output("git ls-remote --tags %s | awk '{print $2}' "
                        "| grep -v '{}' | awk -F\"/\" '{print $3}' "
                        "| sort -n -t. -k1,1 -k2,2 -k3,3 | tail -n 1"
                        % repository_url, shell=True).strip()

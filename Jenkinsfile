def branchName = "${env.BRANCH_NAME}"
def packagePath = "nlu_metrics"
def VENV = ". venv/bin/activate"


def version(path) {
    readFile("${path}/__version__").split("\n")[0]
}


node('jenkins-slave-ec2') {
    stage('Checkout') {
        deleteDir()
        checkout scm
    }

    stage('Setup') {
    	sh "virtualenv venv"
    	def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
    	sh """
    	${VENV}
    	echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
    	pip install .
    	"""
    }

    stage('Tests') {
        sh """
        ${VENV}
        python -m unittest discover
        """
    }

    stage('Publish') {
        switch (branchName) {
            case "master":
                deleteDir()
                checkout scm
                def rootPath = pwd()
                def path = "${rootPath}/${packagePath}"
                def credentials = "${env.NEXUS_USERNAME_PYPI}:${env.NEXUS_PASSWORD_PYPI}"
                sh """
                virtualenv venv
                ${VENV}
                echo "[global]\nindex = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/pypi\nindex-url = https://pypi.python.org/simple/\nextra-index-url = https://${credentials}@nexus-repository.snips.ai/repository/pypi-internal/simple" >> venv/pip.conf
                pip install .
                python setup.py bdist_wheel upload -r pypisnips
                git tag ${version(path)}
                git remote rm origin
                git remote add origin 'git@github.com:snipsco/nlu-metrics.git'
                git config --global user.email 'jenkins@snips.ai'
                git config --global user.name 'Jenkins'
                git push --tags
                """
            default:
                sh """
                ${VENV}
                python setup.py bdist_wheel
                """
        }
    }
}

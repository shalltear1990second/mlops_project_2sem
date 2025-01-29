pipeline{
    agent any

    stages{
        stage('Prepare work folders'){
            steps{
                sh 'mkdir -p ./data_raw_source'
                sh 'mkdir -p ./data_train'
                sh 'mkdir -p ./data_test'
            }
        }
    	stage('Install requirements.txt'){
            steps{
                sh 'pip3 install --no-cache-dir -r requirements.txt'
            }
        }
        stage('Data load from Yandex Disk'){
            steps{
                sh 'python3 data_load.py'
            }
        }
        stage('Data creation'){
            steps{
                sh 'python3 data_creation.py'
            }
        }
        stage('Data preprocessing'){
            steps{
                sh 'python3 model_preprocessing.py'
            }
            
        }
        stage('Model training'){
            steps{
                sh 'python3 model_preparation.py'
            }
        }
        stage('Model testing'){
            steps{
                sh 'python3 model_testing.py'
            }
        }
    }
    post{
        always{
            sh 'echo "All done!"'
        }
    }
}

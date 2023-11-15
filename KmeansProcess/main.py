from KMeanServices import KMeansService
import pandas as pd
import os

class MainApplication:
    def __init__(self, file_path):
        self.file_path = file_path
        self.service = KMeansService()

    def upload_file(self):
        df = pd.read_csv(self.file_path)
        return df

    def process_file(self):
        df = self.upload_file()
        df_numerical = df[['Species', 'PFG']]
        df_clustered = self.service.fit_predict(df_numerical)
        self.service.plot_clusters(df_clustered)

        silhouette_score = self.service.get_silhouette_score(df_clustered)

        davies_bouldin_score = self.service.get_davies_bouldin_score(df_clustered)
        
        calinski_harabasz_score = self.service.get_calinski_harabasz_score(df_clustered)

        self.save_scores_to_csv('KMeans', silhouette_score, davies_bouldin_score, calinski_harabasz_score)



    def save_scores_to_csv(self, model_name, silhouette_score, davies_bouldin_score, calinski_harabasz_score):

        if not os.path.exists('output'):
            os.makedirs('output')
        
        filename = f'output/{model_name}_performance.csv'
        
        scores_df = pd.DataFrame({
            'Model': [model_name],
            'Silhouette Score': [silhouette_score],
            'Davies-Bouldin Score': [davies_bouldin_score],
            'Calinski-Harabasz Score': [calinski_harabasz_score]
        })

        scores_df.to_csv(filename, index=False)
        print(f"Scores have been saved to {filename}")
                

if __name__ == "__main__":
    file_path = 'data\PFG.csv' 
    app = MainApplication(file_path)
    app.process_file()

from utils import *
from data import *
from noise_scheduler import *
from backward_process import *
from training import *

IMG_SIZE = 64
BATCH_SIZE = 256



if __name__ == "__main__":
    # Define the model
    model = SimpleUnet()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Define the dataset
    dataset = load_transformed_dataset()
    # Define the pipeline
    pipeline = DiffusionPipeline(dataset, model, optimizer, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=BATCH_SIZE)
    #run an example
    pipeline.load_state("best_model.pth")
    pipeline.visual_validation(10, 64, num_processes=5)

    #run training
    pipeline.train(200)
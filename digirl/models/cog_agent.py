import signal
from gradio_client import Client
import gradio_client
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

def _get_a_action(pair):
    client, obs = pair
    text = f'What steps do I need to take to "{obs["task"]}"?(with grounding)'
    for _ in range(3):
        try:
            out = client.predict(text, gradio_client.file(obs['image_path']))
            return out
        except:
            sleep(1)
    return None

class CogAgent:
    def __init__(self, url):
        urls = url
        self.clients = [Client(u) for u in urls]
    
    def prepare(self):
        pass
    
    def get_action(self, observation, image_features):
        results = []
        client_obs_pairs = zip(self.clients, observation)
        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            future_to_client_obs = {executor.submit(_get_a_action, pair): pair for pair in client_obs_pairs}
            for future in as_completed(future_to_client_obs):
                # try:
                result = future.result()
                results.append(result)
                # except Exception as exc:
                #     print(f'Generated an exception: {exc}')
        return results


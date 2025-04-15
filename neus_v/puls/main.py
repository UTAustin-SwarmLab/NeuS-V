from puls import *

import argparse
import json
import time

def main():
    prompt = "a dog sits on a mat while a ball rolls past the dog towards a couch, then the dog picks up the ball and places it beside the couch"
    modes = [Mode.OBJECT_ACTION_ALIGNMENT, Mode.OVERALL_CONSISTENCY, Mode.OBJECT_EXISTENCE, Mode.SPATIAL_RELATIONSHIP]

    parser = argparse.ArgumentParser(description='Set OpenAI API Key.')
    parser.add_argument('--openai_key', type=str, help='Your OpenAI API key')
    args = parser.parse_args()
    
    start_time = time.time()
    if args.openai_key:
        data = PULS(prompt, modes, args.openai_key)
    else:
        data = PULS(prompt, modes)
    end_time = time.time()

    print(prompt)
    print(json.dumps(data, indent=2))
    print(f"Elapsed Time: {end_time - start_time}")
    
if __name__ == "__main__":
    main()


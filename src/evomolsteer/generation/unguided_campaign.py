"""Independent unguided collections using the shared progress/archive controller."""
from pathlib import Path

from .steer_campaign import main as campaign_main


def main(argv=None):
    config=Path(__file__).resolve().parents[3]/'configs/generation_hiqbind_unguided.json'
    return campaign_main(argv,required_arm='unguided',default_config=str(config))


if __name__=='__main__':main()

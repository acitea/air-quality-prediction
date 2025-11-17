import time, hopsworks, datetime
import pandas as pd
from typing import List
from rich.console import Console
console = Console()


class Hopsworks:

    def __init__(self):
        from hsfs.feature import Feature
        import os
        self.Feature = Feature
        print("\nConnecting to Hopsworks...")
        self.project = hopsworks.login(engine="python", project=os.getenv("HOPSWORKS_PROJECT_NAME"), api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        self.fs = self.project.get_feature_store()

    def retrieve_from_hopsworks_from_time(self,
        feature_group: str,
        feature_names: List[str],
        version: int,
        time_threshold: datetime
    ) -> pd.DataFrame:
        fg = self.fs.get_feature_group(feature_group, version=version)

        df = fg.select(feature_names) \
            .filter(self.Feature('timestamp', 'timestamp') >= time_threshold) \
            .read()

        return df

    def append_to_hopsworks(self,
        data: dict[str, pd.DataFrame],
        version: int
    ):
        """
        Append dataframes to Hopsworks feature groups

        Args:
            data: Dictionary of dataframes to upload. Keys are: Feature Group Name, and Values are: DataFrame
        """

        idx = 0
        print("\nUploading to Hopsworks feature store...")
        for fg_name, df in data.items():
            if df.empty:
                print(f"  Skipping empty feature group: {fg_name}")
                continue

            print(f"  Uploading to feature group: {fg_name} ({len(df)} records)")

            try:
                if idx == 5: 
                    print("    Pausing for 150 seconds to avoid rate limits...")
                    time.sleep(150)
                    idx = 0
                self.fs.get_feature_group(fg_name, version=version).insert(df)
                idx += 1
            except Exception as e:
                print(f"    ✗ Failed to upload to {fg_name}: {e}")
                console.print_exception()
                raise

        print("\n✓ All data uploaded successfully!")

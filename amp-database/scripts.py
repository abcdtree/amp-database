import json
import pandas as pd
import numpy as np
from scipy.stats import zscore
import argparse
from pathlib import Path


def read_json(mfile):
    with open(mfile, "r") as f:
        data = json.load(f)
    return data


def load_artic_coverage(mfile):
    data = read_json(mfile)
    df = pd.DataFrame()
    for data_dict in data["data"]:
        run_id = data_dict["barcode"]
        coverage_data = data_dict["coverage"]
        columns = ["start", "end", "col1", "col2", "depth_r1", "depth_r2"]
        df_tmp = pd.DataFrame(data=coverage_data, columns=columns)
        df_tmp["sample"] = run_id
        df = pd.concat([df, df_tmp])
        df["depth"] = df["depth_r1"] + df["depth_r2"]
    return df[["sample", "start", "end", "depth_r1", "depth_r2", "depth"]].copy()


def check_sample_df(x):
    if x["mean_depth"] < 20 or x["pass_coverage"] < 90:
        return "FAIL"
    else:
        return "PASS"


def create_sample_df(data_df, threshold=20):
    sample_df_title = [
        "sample",
        "mean_depth",
        "pass_coverage",
        "std_depth",
        "cv_depth",
        "Q1",
        "Q01",
    ]
    sample_df_data = []
    for sample in data_df["sample"].unique().tolist():
        df_sub = data_df[data_df["sample"] == sample]
        mean_depth = df_sub.depth.mean()
        pass_ratio = 100 * (df_sub.depth >= threshold).sum() / len(df_sub.depth)
        # add cv -- Coefficient of Variation
        std = df_sub.depth.std()
        cv = round(mean_depth / std, 3) * 100
        q1 = round(df_sub.depth.quantile(0.25))
        qplus = round(df_sub.depth.quantile(0.1))
        sample_df_data.append([sample, mean_depth, pass_ratio, std, cv, q1, qplus])
    sample_df = pd.DataFrame(data=sample_df_data, columns=sample_df_title)
    sample_df["depth_check"] = sample_df.apply(check_sample_df, axis=1)
    return sample_df


def overlap(a, b, x, y):
    # print(a, b, x, y)
    return a <= y and x <= b


def df_overlap(x, lower, upper):
    return overlap(x["start"], x["end"], lower, upper)


def modified_z_score(data_series):
    median_val = data_series.median()
    # Calculate Median Absolute Deviation (MAD)
    mad = np.median(np.abs(data_series - median_val))

    # Check if MAD is zero to avoid division error
    if mad == 0:
        return pd.Series(0, index=data_series.index)

    # Calculate modified Z-score
    mod_z_scores = 0.6745 * (data_series - median_val) / mad
    return mod_z_scores


def load_scheme(scheme_path):
    scheme_df = pd.read_csv(scheme_path, sep="\t", header=None)
    scheme_df.columns = [
        "Chrome",
        "start",
        "end",
        "pos_id",
        "patch",
        "direction",
        "seq",
    ]
    scheme_section = []
    new_row = []
    for index, x in scheme_df.iterrows():
        if x["direction"] == "+":
            new_row = []
            new_row.append(x["pos_id"].replace("_LEFT_1", ""))
            new_row.append(x["end"])
        else:
            new_row.append(x["start"])
            new_row.append(x["patch"])
            scheme_section.append(new_row)
    return scheme_section


def dropout_decision(x):
    if x["zscore_for_mean"] > -3:
        return "Regular"
    elif x["mean"] < 20:
        return "Amplicon_dropout"
    elif x["zscore_for_mean"] > -5:
        return "Highlight, low coverage region"
    else:
        return "Big Warning, super low coverage region"


def stable_decision(x):
    if x["mean"] >= x["Q1"]:
        return "stable"
    elif x["mean"] >= x["Q01"]:
        return "potential dropout"
    else:
        return "dropout"


def sample_zscore_summary(coverage_df, scheme_section, sample_df):
    title = [
        "sample",
        "amplicon_id",
        "mean",
        "std",
        "median",
        "position_zscore_min",
        "position_zscore_outliers",
        "position_modz_min",
        "position_modz_outliers",
    ]
    sample_list = coverage_df["sample"].unique().tolist()
    summary_data = []
    for s in sample_list:
        df_sub = coverage_df[coverage_df["sample"] == s].copy()
        for row in scheme_section:
            scheme_id = row[0]
            scheme_start = row[1]
            scheme_end = row[2]
            round_n = row[3]
            depth_col = f"depth_r{round_n}"
            in_range = df_sub.apply(
                df_overlap, lower=scheme_start, upper=scheme_end, axis=1
            )
            df_in_range = df_sub[in_range]
            mean = df_in_range[depth_col].mean()
            std = df_in_range[depth_col].std()
            median = df_in_range[depth_col].median()
            mod_z = modified_z_score(df_in_range[depth_col])
            z_score = zscore(df_in_range[depth_col])
            drop_or_spike_z = z_score < -3
            drop_or_spike_z_count = drop_or_spike_z.sum()
            drop_or_spike = mod_z < -3.5
            drop_or_spike_count = drop_or_spike.sum()
            # print(s, scheme_id, mean, std, median, mod_z.max(), mod_z.min(), f"{drop_or_spike_count}/{len(mod_z)}")
            summary_data.append(
                [
                    s,
                    scheme_id,
                    mean,
                    std,
                    median,
                    z_score.min(),
                    f"{drop_or_spike_z_count}/{len(z_score)}",
                    mod_z.min(),
                    f"{drop_or_spike_count}/{len(mod_z)}",
                ]
            )
    amp_df = pd.DataFrame(data=summary_data, columns=title)
    amp_df = pd.merge(amp_df, sample_df, on="sample", how="left")
    amp_df["zscore_for_mean"] = amp_df.groupby(by="sample")["mean"].transform(zscore)
    amp_df["dropout"] = amp_df.apply(dropout_decision, axis=1)
    amp_df["stable"] = amp_df.apply(stable_decision, axis=1)
    # amp_df = pd.merge(amp_df, sample_df, on="sample", how="left")
    return amp_df


def main():
    parser = argparse.ArgumentParser(description="prepare dropout database")
    parser.add_argument("artic", help="artic.json file from the covid analysis")
    parser.add_argument(
        "--folder",
        action="store_true",
        help="artic path is a folder contains a list of sars run to loop",
    )
    # parser.add_argument("--dirtyinput", help="input samplesheet for dirty pipeline to run")
    parser.add_argument("scheme", help="artic scheme bed file")
    parser.add_argument(
        "--threshold", default=20, help="coverage thershold, default as 20X"
    )
    parser.add_argument(
        "--output", default="amp_df.csv", help="csv output to record the data"
    )
    args = parser.parse_args()
    if args.folder:
        scheme_section = load_scheme(args.scheme)
        artic_folder = Path(args.artic)
        summary_df = pd.DataFrame()
        for item in artic_folder.rglob("artic.json"):
            print(f"-- Working on file {item} -- ")
            run_id = item.parts[-3]
            coverage_df = load_artic_coverage(item)
            sample_df = create_sample_df(coverage_df, args.threshold)
            amp_df = sample_zscore_summary(coverage_df, scheme_section, sample_df)
            amp_df["run_id"] = run_id
            amp_df.to_csv(f"{run_id}_amp_df.csv", index=None)
            summary_df = pd.concat([summary_df, amp_df], ignore_index=True)
            print(f"-- Results for {run_id} are saved to local file --")
        summary_df.to_csv("full_summary.csv", index=None)

        # concat the result.tsv as well
        result_df = pd.DataFrame()
        print("merge the result tsv files")
        for item in artic_folder.rglob("*_results.tsv"):
            df_tmp = pd.read_csv(item, sep="\t")
            result_df = pd.concat([result_df, df_tmp], ignore_index=True)
        result_df.to_csv("artic_result.csv", index=None)
    else:
        coverage_df = load_artic_coverage(args.artic)
        sample_df = create_sample_df(coverage_df, args.threshold)
        scheme_section = load_scheme(args.scheme)
        # print(scheme_section)
        amp_df = sample_zscore_summary(coverage_df, scheme_section, sample_df)
        amp_df.to_csv(args.output, index=None)


if __name__ == "__main__":
    main()

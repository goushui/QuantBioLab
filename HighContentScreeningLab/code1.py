import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, filters, measure, morphology
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
from scipy.spatial.distance import pdist

np.random.seed(67)

def parse_filename(fname):
    """Extract metadata from filename: G{group}_{mag}_{well}_{field}_T{treatment number}_{channel}.jpg"""
    components = fname.replace('.jpg', '').split('_')
    treatment_id = components[4]
    print("Treatment_id: "+treatment_id)
    day_id = 1 if int(components[0][1])<=6 else 2
    return {
        'filename': fname,
        'group': components[0] if len(components) > 0 else 'Unknown',
        'treatment': treatment_id[1:] if treatment_id.startswith('T') else 'Unknown',
        'day': day_id
    }

def load_images(img_dir):
    """Load all images for specified channel"""
    feat_list = []
    meta_list = []
    # get unique names for _DAPI and _TRANS channels
    unique_base_names = set()
    count = 0
    for fname in os.listdir(img_dir):
        # remove the channel part to get unique base names
        base_name = "_".join(fname.split('_')[:-1])
        unique_base_names.add(base_name)
    print("Number of samples: " + str(len(unique_base_names)))
    for name in unique_base_names:
        print(str(count)+" processing DAPI and TRANS images for "+name)
        count += 1
        meta = parse_filename(name)
        # for DAPI
        fpath = os.path.join(img_dir, name + '_DAPI.jpg')
        try:
            dapi_img = io.imread(fpath)
            features_dapi = extract_features_DAPI(dapi_img)
        except Exception as e:
            print("Error loading DAPI image for {}: {}".format(name, e))
            continue
        # for TRANS
        
        fpath = os.path.join(img_dir, name + '_TRANS.jpg')
        try:
            trans_img = io.imread(fpath)
            features_trans = extract_features_TRANS(trans_img)
        except Exception as e:
            print("Error loading TRANS image for {}: {}".format(name, e))
            continue
        print("extracting features")
        print("finished extracting features appending meta")
        meta_list.append(meta)
        feat_list.append({**features_dapi, **features_trans})
        print("finished appending meta")
    return feat_list, meta_list

def extract_features_DAPI(img):
    """Extract 5 types of features: texture, morphology, intensity, spatial, edge"""
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
    feat_dict = {}
    # TYPE 2: Morphology - shape and size of nuclei
    try:
        print("Beginning image proc")
        thresh_val = threshold_otsu(img)
        binary_img = morphology.remove_small_objects(img > thresh_val, min_size=50)
        regions = measure.regionprops(measure.label(binary_img))
        print("Finished Image process")
        if len(regions) > 0:
            feat_dict['mean_nucleus_area'] = np.mean([r.area for r in regions])
            feat_dict['std_nucleus_area'] = np.std([r.area for r in regions])
            # feat_dict['mean_solidity'] = np.mean([r.solidity for r in regions])
            feat_dict['mean_eccentricity'] = np.mean([r.eccentricity for r in regions])
            feat_dict['num_objects'] = len(regions)
        else:
            feat_dict['mean_nucleus_area'] = feat_dict['std_nucleus_area'] = 0
            feat_dict['mean_solidity'] = feat_dict['mean_eccentricity'] = 0
            feat_dict['num_objects'] = 0
    except:
        feat_dict['mean_nucleus_area'] = feat_dict['std_nucleus_area'] = 0
        feat_dict['mean_solidity'] = feat_dict['mean_eccentricity'] = 0
        feat_dict['num_objects'] = 0

    # TYPE 4: Spatial distribution - cell arrangement
    try:
        if feat_dict['num_objects'] > 2:
            centers = np.array([r.centroid for r in regions])
            dists = pdist(centers)
            feat_dict['mean_nn_distance'] = np.mean(dists)
            feat_dict['std_nn_distance'] = np.std(dists)
        else:
            feat_dict['mean_nn_distance'] = feat_dict['std_nn_distance'] = 0
    except:
        feat_dict['mean_nn_distance'] = feat_dict['std_nn_distance'] = 0

    return feat_dict

def extract_features_TRANS(img):
    """Extract 5 types of features: texture, morphology, intensity, spatial, edge"""
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
    feat_dict = {}

    # TYPE 3: Intensity - brightness characteristics
    feat_dict['mean_intensity'] = np.mean(img)

    # TYPE 5: Edges - nuclear boundaries
    edge_map = filters.sobel(img)
    feat_dict['edge_mean'] = np.mean(edge_map)
    feat_dict['edge_std'] = np.std(edge_map)

    return feat_dict

def train_classifier(feature_matrix, labels, task_desc, out_dir, filenames, min_samples=2):
    """Train RandomForest classifier and save results"""
    label_counts = pd.Series(labels).value_counts()
    valid_labels = label_counts[label_counts >= min_samples].index
    label_mask = pd.Series(labels).isin(valid_labels)
    feature_matrix, labels = feature_matrix[label_mask], labels[label_mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_te)
    accuracy = accuracy_score(y_te, predictions)

    # Feature importance plot
    importance_df = pd.DataFrame({'feature': feature_matrix.columns, 'importance': model.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False)

    num_feat = min(len(importance_df), 20)
    plt.figure(figsize=(10, max(6, num_feat * 0.4)))
    plt.barh(range(num_feat), importance_df.head(num_feat)['importance'])
    plt.yticks(range(num_feat), importance_df.head(num_feat)['feature'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {task_desc}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"importance_{task_desc.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()

    # Confusion matrix plot
    conf_mat = confusion_matrix(y_te, predictions)
    conf_mat_norm = conf_mat.astype('float') / (conf_mat.sum(axis=1)[:, np.newaxis] + 1e-10)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=sorted(valid_labels), yticklabels=sorted(valid_labels))
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {task_desc}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_{task_desc.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()

    # Save misclassifications
    errors_df = pd.DataFrame({
        'filename': [filenames[y_te.index[i]] for i in range(len(y_te)) if y_te.iloc[i] != predictions[i]],
        'true': [y_te.iloc[i] for i in range(len(y_te)) if y_te.iloc[i] != predictions[i]],
        'predicted': [predictions[i] for i in range(len(y_te)) if y_te.iloc[i] != predictions[i]]
    })
    if len(errors_df) > 0:
        errors_df.to_csv(os.path.join(out_dir, f"misclass_{task_desc.lower().replace(' ', '_')}.csv"), index=False)

    return accuracy

def main():
    img_dir = "./Images"
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    feat_list, meta_list = load_images(img_dir)

    feat_df = pd.DataFrame(feat_list)
    meta_df = pd.DataFrame(meta_list)

    combined_df = pd.concat([meta_df, feat_df], axis=1)
    combined_df.to_csv(os.path.join(out_dir, "features.csv"), index=False)

    filenames = os.listdir(img_dir)
    acc_treatment = train_classifier(feat_df, meta_df['treatment'], 'Treatment Prediction', out_dir, filenames)
    acc_group = train_classifier(feat_df, meta_df['group'], 'Group Prediction', out_dir, filenames)
    acc_day = train_classifier(feat_df, meta_df['day'], 'Day Prediction', out_dir, filenames)

if __name__ == "__main__":
    main()

# data_link: https://www.kaggle.com/code/fedesoriano/spanish-wine-quality-dataset-introduction/notebook?select=wines_SPA.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pydotplus
import warnings
warnings.filterwarnings('ignore')

def veri_on_isleme(df):
    # Eksik değerleri kontrol et ve temizle
    print("\nEksik Değer Analizi:")
    print(df.isnull().sum())
    
    # 'N.V.' değerlerini medyan ile değiştir
    df['year'] = pd.to_numeric(df['year'].replace('N.V.', np.nan), errors='coerce')
    df['year'] = df['year'].fillna(df['year'].median())
    
    # Eksik değerleri doldur
    df['body'] = df['body'].fillna(df['body'].median())
    df['acidity'] = df['acidity'].fillna(df['acidity'].median())
    df['type'] = df['type'].fillna('Unknown')
    
    print("\nRating Dağılımı:")
    print(df['rating'].describe())
    
    # Rating değerlerini kategorilere ayır
    df['rating_category'] = pd.cut(df['rating'], 
                                 bins=[4.1, 4.3, 4.5, 4.7, 5.0],
                                 labels=['İyi', 'Çok İyi', 'Mükemmel', 'Üstün'])
    
    print("\nRating Kategorileri Dağılımı:")
    print(df['rating_category'].value_counts())
    
    # Kategori eşleştirmelerini güncelle
    rating_mapping = {
        'İyi': 0,          # En düşük kategori -> 0
        'Çok İyi': 1,      # İkinci kategori -> 1
        'Mükemmel': 2,     # Üçüncü kategori -> 2
        'Üstün': 3         # En yüksek kategori -> 3
    }

    # Kategorileri yeni numaralarla güncelle
    df['rating_category'] = df['rating_category'].map(rating_mapping)
    
    # Kategorik değişkenleri sayısallaştır
    le = LabelEncoder()
    kategorik_kolonlar = ['winery', 'wine', 'country', 'region', 'type']
    for kolon in kategorik_kolonlar:
        df[kolon] = le.fit_transform(df[kolon])
    
    # Rating kategorilerini sayısallaştır ama etiketleri sakla
    rating_le = LabelEncoder()
    df['rating_category_encoded'] = rating_le.fit_transform(df['rating_category'])
    
    print("\nRating Kategorileri Eşleştirmesi:")
    category_mapping = {
        0: "İyi",
        1: "Çok İyi",
        2: "Mükemmel",
        3: "Üstün"
    }

    # Kategorileri yazdır
    for code, category in category_mapping.items():
        print(f"{code}: {category}")
    
    return df, rating_le

def plot_learning_curve(model, model_name, X, y):
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=train_sizes,
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Eğitim skoru', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, label='Çapraz doğrulama skoru', color='green', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
    
    plt.xlabel('Eğitim örnekleri sayısı')
    plt.ylabel('Doğruluk')
    plt.title(f'Öğrenme Eğrisi - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'output_png/learning_curve_{model_name}.png')
    plt.close()

def derinlik_analizi(X_train, X_test, y_train, y_test):
    derinlikler = range(1, 21)
    train_scores = []
    test_scores = []
    
    for derinlik in derinlikler:
        dt = DecisionTreeClassifier(max_depth=derinlik, random_state=42)
        dt.fit(X_train, y_train)
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(derinlikler, train_scores, label='Eğitim skoru', marker='o')
    plt.plot(derinlikler, test_scores, label='Test skoru', marker='s')
    plt.xlabel('Ağaç Derinliği')
    plt.ylabel('Doğruluk')
    plt.title('Karar Ağacı Derinlik Analizi')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_png/depth_analysis.png')
    plt.close()
    
    en_iyi_derinlik = derinlikler[np.argmax(test_scores)]
    print(f"En iyi test doğruluğu:")
    print(f"Derinlik: {en_iyi_derinlik}, Doğruluk: {test_scores[en_iyi_derinlik-1]:.4f}")
    
    return en_iyi_derinlik

def gradient_boosting_analizi(X_train, X_test, y_train, y_test):
    # Normal GBM
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbm.fit(X_train, y_train)
    train_score = gbm.score(X_train, y_train)
    test_score = gbm.score(X_test, y_test)
    
    print("Gradient Boosting Sonuçları:")
    print(f"Eğitim doğruluğu: {train_score:.4f}")
    print(f"Test doğruluğu: {test_score:.4f}")
    
    plot_learning_curve(gbm, "GradientBoosting", X_train, y_train)
    
    # Overfitting GBM
    gbm_overfit = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=7, random_state=42)
    gbm_overfit.fit(X_train, y_train)
    train_score = gbm_overfit.score(X_train, y_train)
    test_score = gbm_overfit.score(X_test, y_test)
    
    print("\nOverfitting GBM Sonuçları:")
    print(f"Eğitim doğruluğu: {train_score:.4f}")
    print(f"Test doğruluğu: {test_score:.4f}")
    
    plot_learning_curve(gbm_overfit, "Overfitted_GradientBoosting", X_train, y_train)

def adaboost_analizi(X_train, X_test, y_train, y_test):
    # Normal AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    ada.fit(X_train, y_train)
    train_score = ada.score(X_train, y_train)
    test_score = ada.score(X_test, y_test)
    
    print("AdaBoost Sonuçları:")
    print(f"Eğitim doğruluğu: {train_score:.4f}")
    print(f"Test doğruluğu: {test_score:.4f}")
    
    plot_learning_curve(ada, "AdaBoost", X_train, y_train)
    
    # Aşırı öğrenme için AdaBoost
    ada_overfit = AdaBoostClassifier(
        n_estimators=500,  # Daha fazla ağaç
        learning_rate=1.0, # Daha yüksek öğrenme oranı
        random_state=42
    )
    ada_overfit.fit(X_train, y_train)
    train_score = ada_overfit.score(X_train, y_train)
    test_score = ada_overfit.score(X_test, y_test)
    
    print("\nOverfitting AdaBoost Sonuçları:")
    print(f"Eğitim doğruluğu: {train_score:.4f}")
    print(f"Test doğruluğu: {test_score:.4f}")
    
    plot_learning_curve(ada_overfit, "Overfitted_AdaBoost", X_train, y_train)

def karar_agaci_analizi(X_train, X_test, y_train, y_test, target_names):  
    target_names = ['İyi', 'Çok İyi', 'Mükemmel', 'Üstün']
    
    # Farklı derinliklerle karar ağacı modelleri
    derinlikler = [3, 5, 7, 10]
    best_score = 0
    best_depth = None
    best_model = None
    
    for derinlik in derinlikler:
        dt = DecisionTreeClassifier(max_depth=derinlik, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        
        # En iyi modeli sakla
        if test_score > best_score:
            best_score = test_score
            best_depth = derinlik
            best_model = dt
        
        print(f"\nDerinlik {derinlik}:")
        print(f"Eğitim doğruluğu: {train_score:.4f}")
        print(f"Test doğruluğu: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
        print(f"5-fold CV ortalama doğruluk: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # En iyi model için detaylı metrikler
    print(f"\nEn iyi model (Derinlik {best_depth}):")
    y_pred = best_model.predict(X_test)
    
    # Confusion Matrix ve Classification Report
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Karmaşıklık matrisini görselleştir
    plt = plot_confusion_matrix(cm, f'Karar Ağacı (Derinlik={best_depth}) Karmaşıklık Matrisi', target_names)
    plt.savefig('output_png/karar_agaci_confusion_matrix.png')
    plt.close()
    
    # Karar ağacı görselleştirmesi
    feature_names = ['winery', 'wine', 'year', 'num_reviews', 'country', 
                    'region', 'price', 'type', 'body', 'acidity']
    
    dot_data = export_graphviz(best_model, 
                             out_file='output_png/tree.dot',
                             feature_names=feature_names,
                             class_names=target_names,
                             filled=True,
                             rounded=True,
                             special_characters=True)
    
    import subprocess
    try:
        dot_path = '/opt/homebrew/bin/dot'
        subprocess.run([dot_path, '-Tpng', 'output_png/tree.dot', '-o', 'output_png/karar_agaci.png'], 
                     check=True)
        print("\nKarar ağacı görselleştirmesi 'output_png/karar_agaci.png' dosyasına kaydedildi.")
    except Exception as e:
        print(f"Graphviz hatası: {e}")

def plot_confusion_matrix(cm, title, target_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    # Sayıları matris üzerine yerleştir
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    return plt

def knn_analizi(X_train, X_test, y_train, y_test):  
    target_names = ['İyi', 'Çok İyi', 'Mükemmel', 'Üstün']
    k_values = [3, 5, 7, 9]
    best_score = 0
    best_k = None
    best_model = None
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        score = knn.score(X_test, y_test)
        print(f"\nk={k} için sonuçlar:")
        print(f"Test doğruluğu: {score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
        print(f"5-fold CV ortalama doğruluk: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # En iyi modeli sakla
        if score > best_score:
            best_score = score
            best_k = k
            best_model = knn
    
    # En iyi model için detaylı metrikler
    print(f"\nEn iyi KNN modeli (k={best_k}):")
    y_pred = best_model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nKarmaşıklık Matrisi:")
    print(cm)
    
    # Karmaşıklık matrisini görselleştir
    plt = plot_confusion_matrix(cm, f'KNN (k={best_k}) Karmaşıklık Matrisi', target_names)
    plt.savefig('output_png/knn_confusion_matrix.png')
    plt.close()
    
    # Detaylı metrik analizi
    print("\nDetaylı Metrik Analizi:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

def svm_analizi(X_train, X_test, y_train, y_test):
    target_names = ['İyi', 'Çok İyi', 'Mükemmel', 'Üstün']
    kernels = ['linear', 'rbf']
    best_score = 0
    best_kernel = None
    best_model = None
    
    for kernel in kernels:
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train, y_train)
        
        score = svm.score(X_test, y_test)
        print(f"\n{kernel.upper()} kernel sonuçları:")
        print(f"Test doğruluğu: {score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
        print(f"5-fold CV ortalama doğruluk: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # En iyi modeli sakla
        if score > best_score:
            best_score = score
            best_kernel = kernel
            best_model = svm
    
    # En iyi model için detaylı metrikler
    print(f"\nEn iyi SVM modeli ({best_kernel} kernel):")
    y_pred = best_model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nKarmaşıklık Matrisi:")
    print(cm)
    
    # Karmaşıklık matrisini görselleştir
    plt = plot_confusion_matrix(cm, f'SVM ({best_kernel}) Karmaşıklık Matrisi', target_names)
    plt.savefig('output_png/svm_confusion_matrix.png')
    plt.close()
    
    # Detaylı metrik analizi
    print("\nDetaylı Metrik Analizi:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

def main():
    try:
        # Veri okuma
        df = pd.read_csv("data/wines_SPA.csv")
        print("Veri seti boyutu:", df.shape)
        
        # Veri ön işleme
        df, rating_le = veri_on_isleme(df)
        
        # Özellik ve hedef değişkenleri ayır
        X = df.drop(['rating', 'rating_category', 'rating_category_encoded'], axis=1)
        y = df['rating_category_encoded']
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Özellik ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n=== Derinlik Analizi ===")
        derinlik_analizi(X_train_scaled, X_test_scaled, y_train, y_test)
        
        print("\n=== Gradient Boosting Analizi ===")
        gradient_boosting_analizi(X_train_scaled, X_test_scaled, y_train, y_test)
        
        print("\n=== AdaBoost Analizi ===")
        adaboost_analizi(X_train_scaled, X_test_scaled, y_train, y_test)
        
        print("\n=== Karar Ağacı Analizi ===")
        karar_agaci_analizi(X_train_scaled, X_test_scaled, y_train, y_test, rating_le.classes_)
        
        print("\n=== KNN Analizi ===")
        knn_analizi(X_train_scaled, X_test_scaled, y_train, y_test)
        
        print("\n=== SVM Analizi ===")
        svm_analizi(X_train_scaled, X_test_scaled, y_train, y_test)
            
    except Exception as e:
        print(f"Ana program hatası: {e}")

if __name__ == "__main__":
    main()

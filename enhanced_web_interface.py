import json
import os
from glob import glob
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import procrustes
from plotly.subplots import make_subplots
from sklearn.manifold import MDS

# Global variables to store models and data
bible_model = None
quran_model = None
word_data = []
coords = []
labels = []
colors = []

def load_and_train_models():
    """Load and train separate models for Bible and Quran"""
    global bible_model, quran_model, word_data, coords, labels, colors
    
    # Paths - 使用合并后的古兰经文件
    quran_file = "quran/quran_text_only.json"
    bible_path = "bible"
    
    # Collect sentences separately
    bible_sentences = []
    quran_sentences = []
    
    # Collect Quran sentences from merged file
    try:
        with open(quran_file, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        
        for verse in quran_data['verses']:
            quran_sentences.append(verse['text'].lower().split())
        
        print(f"Loaded {len(quran_data['verses'])} Quran verses")
    except Exception as e:
        print(f"Error loading Quran file: {e}")
        # Fallback to original method
        quran_pattern = "quran/en/en_translation_*.json"
        for fn in glob(quran_pattern):
            try:
                with open(fn, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                verses = data['verse']
                for v in verses.values():
                    quran_sentences.append(v.lower().split())
            except:
                continue
    
    # Collect Bible sentences
    for fn in glob(os.path.join(bible_path, "*.json")):
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for chap in data['chapters']:
                for verse in chap['verses']:
                    bible_sentences.append(verse['text'].lower().split())
        except:
            continue
    
    print(f"Bible sentences collected: {len(bible_sentences)}")
    print(f"Quran sentences collected: {len(quran_sentences)}")
    
    # Train separate models
    print("Training Bible model...")
    bible_model = Word2Vec(bible_sentences, vector_size=300, window=15, min_count=2, workers=4, epochs=50, sg=1)
    
    print("Training Quran model...")
    quran_model = Word2Vec(quran_sentences, vector_size=300, window=15, min_count=2, workers=4, epochs=50, sg=1)
    
    return bible_model, quran_model

def analyze_forgiveness_theme(bible_model, quran_model, similarity_threshold=0.5):
    """Analyze forgiveness theme with multiple related keywords"""
    
    # 定义多个相关的宽恕主题词
    forgiveness_keywords = {
        'forgiveness': ['forgiveness', 'forgive', 'forgiven', 'forgiving'],
    }
    
    word_data = []
    coords = []
    labels = []
    colors = []
    
    print(f"\n=== Analyzing Forgiveness Theme ===")
    print(f"Similarity threshold: {similarity_threshold}")
    
    for theme_name, keywords in forgiveness_keywords.items():
        print(f"\n--- {theme_name.upper()} Theme ---")
        
        for keyword in keywords:
            # Check if keyword exists in both models
            if keyword in bible_model.wv and keyword in quran_model.wv:
                print(f"Found '{keyword}' in both models")
                
                # Add theme word from both models
                bible_vector = bible_model.wv[keyword]
                quran_vector = quran_model.wv[keyword]
                
                word_data.extend([
                    {
                        'word': f"{keyword}_bible",
                        'type': 'theme',
                        'source': 'bible',
                        'original_word': keyword,
                        'theme': theme_name,
                        'similarity': 1.0,
                        'vector': bible_vector.tolist()
                    },
                    {
                        'word': f"{keyword}_quran",
                        'type': 'theme',
                        'source': 'quran',
                        'original_word': keyword,
                        'theme': theme_name,
                        'similarity': 1.0,
                        'vector': quran_vector.tolist()
                    }
                ])
                labels.extend([f"Bible:{keyword}", f"Quran:{keyword}"])
                colors.extend(['green', 'green'])
                
                # Find similar words in Bible model
                if keyword in bible_model.wv:
                    bible_similar_words = bible_model.wv.most_similar(keyword, topn=30)
                    for word, sim in bible_similar_words:
                        if sim >= similarity_threshold and word in bible_model.wv:
                            word_data.append({
                                'word': word,
                                'type': 'bible_similar',
                                'source': 'bible',
                                'theme': theme_name,
                                'similarity': float(sim),
                                'vector': bible_model.wv[word].tolist()
                            })
                            labels.append(f"Bible:{word}")
                            colors.append('blue')
                            print(f"  Bible: {word} - {sim:.3f}")
                
                # Find similar words in Quran model
                if keyword in quran_model.wv:
                    quran_similar_words = quran_model.wv.most_similar(keyword, topn=30)
                    for word, sim in quran_similar_words:
                        if sim >= similarity_threshold and word in quran_model.wv:
                            word_data.append({
                                'word': word,
                                'type': 'quran_similar',
                                'source': 'quran',
                                'theme': theme_name,
                                'similarity': float(sim),
                                'vector': quran_model.wv[word].tolist()
                            })
                            labels.append(f"Quran:{word}")
                            colors.append('red')
                            print(f"  Quran: {word} - {sim:.3f}")
    
    # Align the vector spaces using Procrustes
    if len(word_data) >= 6:
        print("\nAligning vector spaces using Procrustes...")
        aligned_vectors = align_vector_spaces([data['vector'] for data in word_data], word_data)
        coords = aligned_vectors
    else:
        print("Warning: Not enough data points for alignment, using PCA")
        if len(word_data) >= 3:
            pca = PCA(n_components=3)
            coords = pca.fit_transform([data['vector'] for data in word_data]).tolist()
        else:
            coords = [[np.random.randn() for _ in range(3)] for _ in range(len(word_data))]
    
    # Debug information
    bible_count = sum(1 for label in labels if label.startswith('Bible:'))
    quran_count = sum(1 for label in labels if label.startswith('Quran:'))
    theme_count = sum(1 for data in word_data if data['type'] == 'theme')
    
    print(f"\n=== Summary ===")
    print(f"Found Bible words: {bible_count}")
    print(f"Found Quran words: {quran_count}")
    print(f"Found theme words: {theme_count}")
    print(f"Total words: {len(word_data)}")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # 验证相似度关系
    print("\n=== Similarity Verification ===")
    if len(word_data) > 1:
        # 找到主题词
        theme_words = [data for data in word_data if data['type'] == 'theme']
        if theme_words:
            theme_word = theme_words[0]
            theme_coord = coords[word_data.index(theme_word)]
            
            # 计算所有词与主题词的3D距离
            distances_3d = []
            similarities_original = []
            
            for i, data in enumerate(word_data):
                if data['type'] != 'theme':
                    coord = coords[i]
                    # 计算3D欧几里得距离
                    distance_3d = np.sqrt(sum((np.array(coord) - np.array(theme_coord))**2))
                    distances_3d.append(distance_3d)
                    similarities_original.append(data['similarity'])
            
            # 计算相关性
            if len(distances_3d) > 1:
                correlation = np.corrcoef(distances_3d, similarities_original)[0, 1]
                print(f"Correlation between 3D distance and Word2Vec similarity: {correlation:.3f}")
                print("(Negative correlation is expected: smaller distance = higher similarity)")
                
                # 显示前5个最相似的词
                sorted_indices = np.argsort(similarities_original)[::-1][:5]
                print("\nTop 5 most similar words (by Word2Vec similarity):")
                for idx in sorted_indices:
                    word_idx = [i for i, data in enumerate(word_data) if data['type'] != 'theme'][idx]
                    word_data_item = word_data[word_idx]
                    print(f"  {word_data_item['word']}: similarity={word_data_item['similarity']:.3f}, 3D_distance={distances_3d[idx]:.3f}")
    
    return word_data, coords, labels, colors

def align_vector_spaces(vectors, word_data):
    """Align two vector spaces using Procrustes analysis with better similarity preservation"""
    
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Separate vectors by source
    bible_vectors = []
    quran_vectors = []
    bible_indices = []
    quran_indices = []
    
    for i, data in enumerate(word_data):
        if data['source'] == 'bible':
            bible_vectors.append(vectors[i])
            bible_indices.append(i)
        elif data['source'] == 'quran':
            quran_vectors.append(vectors[i])
            quran_indices.append(i)
    
    if len(bible_vectors) < 3 or len(quran_vectors) < 3:
        print("Warning: Not enough vectors for alignment, using PCA")
        pca = PCA(n_components=3)
        return pca.fit_transform(vectors).tolist()
    
    # Convert to numpy arrays
    bible_vectors = np.array(bible_vectors)
    quran_vectors = np.array(quran_vectors)
    
    print(f"Bible vectors: {len(bible_vectors)}, Quran vectors: {len(quran_vectors)}")
    
    # 使用相似度矩阵和MDS来保持相似度关系
    all_vectors = np.vstack([bible_vectors, quran_vectors])
    similarity_matrix = cosine_similarity(all_vectors)
    
    print("Computing 3D coordinates using MDS to preserve similarity relationships...")
    
    # 使用MDS将相似度矩阵转换为3D坐标，保持相似度关系
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    # 将相似度转换为距离（1 - similarity）
    distance_matrix = 1 - similarity_matrix
    coords_3d = mds.fit_transform(distance_matrix)
    
    # 分离圣经和古兰经的坐标
    bible_coords = coords_3d[:len(bible_vectors)]
    quran_coords = coords_3d[len(bible_vectors):]
    
    # 找到共同词汇进行对齐
    common_words = []
    bible_common_indices = []
    quran_common_indices = []
    
    # 收集所有词汇
    bible_words = []
    quran_words = []
    
    for i, data in enumerate(word_data):
        if data['source'] == 'bible':
            bible_words.append(data['word'])
        elif data['source'] == 'quran':
            quran_words.append(data['word'])
    
    # 找到共同词汇
    for i, bible_word in enumerate(bible_words):
        if bible_word in quran_words:
            quran_idx = quran_words.index(bible_word)
            common_words.append(bible_word)
            bible_common_indices.append(i)
            quran_common_indices.append(quran_idx)
    
    print(f"Found {len(common_words)} common words for alignment")
    
    if len(common_words) >= 3:
        # 使用共同词汇进行Procrustes对齐
        bible_common_coords = bible_coords[bible_common_indices]
        quran_common_coords = quran_coords[quran_common_indices]
        
        # Procrustes对齐
        bible_aligned, quran_aligned, disparity = procrustes(bible_common_coords, quran_common_coords)
        
        print(f"Procrustes disparity: {disparity:.6f}")
        
        # 计算变换矩阵
        transform_matrix = np.dot(quran_aligned.T, np.linalg.pinv(bible_aligned.T))
        
        # 对所有圣经坐标应用变换
        all_bible_aligned = np.dot(bible_coords, transform_matrix.T)
        all_quran_aligned = quran_coords  # 古兰经坐标保持原样
        
    else:
        # 如果共同词汇太少，使用最小公共子集
        min_size = min(len(bible_coords), len(quran_coords))
        print(f"Using minimum common subset of {min_size} coordinates")
        
        # 取前min_size个坐标
        bible_subset = bible_coords[:min_size]
        quran_subset = quran_coords[:min_size]
        
        # Procrustes对齐
        bible_aligned, quran_aligned, disparity = procrustes(bible_subset, quran_subset)
        
        print(f"Procrustes disparity: {disparity:.6f}")
        
        # 计算变换矩阵
        transform_matrix = np.dot(quran_aligned.T, np.linalg.pinv(bible_aligned.T))
        
        # 对所有坐标应用变换
        all_bible_aligned = np.dot(bible_coords, transform_matrix.T)
        all_quran_aligned = quran_coords
    
    # 重新构建完整的坐标列表
    aligned_coords = []
    bible_idx = 0
    quran_idx = 0
    
    for i, data in enumerate(word_data):
        if data['source'] == 'bible':
            aligned_coords.append(all_bible_aligned[bible_idx].tolist())
            bible_idx += 1
        elif data['source'] == 'quran':
            aligned_coords.append(all_quran_aligned[quran_idx].tolist())
            quran_idx += 1
    
    return aligned_coords

def create_enhanced_visualization():
    """Create enhanced visualization with multiple forgiveness themes"""
    
    print("Loading and training separate models...")
    bible_model, quran_model = load_and_train_models()
    print("Models loaded successfully!")
    
    # Analyze forgiveness theme with multiple keywords
    word_data, coords, labels, colors = analyze_forgiveness_theme(bible_model, quran_model, similarity_threshold=0.5)
    
    # Create subplots: 1 row, 3 columns
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Bible Words Only', 'Quran Words Only', 'Aligned Vector Spaces'),
        horizontal_spacing=0.05
    )
    
    # Color mapping for different themes
    theme_colors = {
        'forgiveness': 'green',
        'mercy': 'purple', 
        'pardon': 'orange',
        'redemption': 'brown',
        'grace': 'pink'
    }
    
    source_colors = {
        'theme': 'gold',
        'bible_similar': 'blue',
        'quran_similar': 'red'
    }
    
    # Separate data by source for individual plots
    bible_data = [data for data in word_data if data['source'] == 'bible']
    quran_data = [data for data in word_data if data['source'] == 'quran']
    
    # Get coordinates for each source
    bible_coords = []
    quran_coords = []
    
    for data in bible_data:
        idx = word_data.index(data)
        if idx < len(coords):
            bible_coords.append(coords[idx])
    
    for data in quran_data:
        idx = word_data.index(data)
        if idx < len(coords):
            quran_coords.append(coords[idx])
    
    # Create Bible-only visualization (left plot)
    if bible_coords:
        for word_type in ['theme', 'bible_similar']:
            mask = [word['type'] == word_type for word in bible_data]
            if any(mask):
                x_coords = [bible_coords[i][0] for i in range(len(bible_coords)) if mask[i]]
                y_coords = [bible_coords[i][1] for i in range(len(bible_coords)) if mask[i]]
                z_coords = [bible_coords[i][2] for i in range(len(bible_coords)) if mask[i]]
                words = [word['word'] for word in bible_data if word['type'] == word_type]
                similarities = [word.get('similarity', 1.0) for word in bible_data if word['type'] == word_type]
                themes = [word.get('theme', 'unknown') for word in bible_data if word['type'] == word_type]
                
                # Create hover text
                hover_text = []
                for word, sim, theme in zip(words, similarities, themes):
                    if word_type == 'theme':
                        hover_text.append(f"Word: {word}<br>Type: Theme Word<br>Theme: {theme}<br>Similarity: {sim:.3f}")
                    else:
                        hover_text.append(f"Word: {word}<br>Type: Similar Word<br>Theme: {theme}<br>Similarity: {sim:.3f}")
                
                # Adjust point size based on similarity
                marker_sizes = [20 + sim * 30 for sim in similarities]
                
                # Use theme colors for theme words, source colors for similar words
                marker_colors = []
                for theme in themes:
                    if word_type == 'theme':
                        marker_colors.append(theme_colors.get(theme, 'gold'))
                    else:
                        marker_colors.append(source_colors[word_type])
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers+text',
                    marker=dict(
                        size=marker_sizes,
                        color=marker_colors,
                        opacity=0.8
                    ),
                    text=words,
                    textposition="middle center",
                    hovertext=hover_text,
                    hoverinfo='text',
                    name=f"Bible {word_type.replace('_', ' ').title()}",
                    showlegend=True
                ), row=1, col=1)
    
    # Create Quran-only visualization (middle plot)
    if quran_coords:
        for word_type in ['theme', 'quran_similar']:
            mask = [word['type'] == word_type for word in quran_data]
            if any(mask):
                x_coords = [quran_coords[i][0] for i in range(len(quran_coords)) if mask[i]]
                y_coords = [quran_coords[i][1] for i in range(len(quran_coords)) if mask[i]]
                z_coords = [quran_coords[i][2] for i in range(len(quran_coords)) if mask[i]]
                words = [word['word'] for word in quran_data if word['type'] == word_type]
                similarities = [word.get('similarity', 1.0) for word in quran_data if word['type'] == word_type]
                themes = [word.get('theme', 'unknown') for word in quran_data if word['type'] == word_type]
                
                # Create hover text
                hover_text = []
                for word, sim, theme in zip(words, similarities, themes):
                    if word_type == 'theme':
                        hover_text.append(f"Word: {word}<br>Type: Theme Word<br>Theme: {theme}<br>Similarity: {sim:.3f}")
                    else:
                        hover_text.append(f"Word: {word}<br>Type: Similar Word<br>Theme: {theme}<br>Similarity: {sim:.3f}")
                
                # Adjust point size based on similarity
                marker_sizes = [20 + sim * 30 for sim in similarities]
                
                # Use theme colors for theme words, source colors for similar words
                marker_colors = []
                for theme in themes:
                    if word_type == 'theme':
                        marker_colors.append(theme_colors.get(theme, 'gold'))
                    else:
                        marker_colors.append(source_colors[word_type])
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers+text',
                    marker=dict(
                        size=marker_sizes,
                        color=marker_colors,
                        opacity=0.8
                    ),
                    text=words,
                    textposition="middle center",
                    hovertext=hover_text,
                    hoverinfo='text',
                    name=f"Quran {word_type.replace('_', ' ').title()}",
                    showlegend=True
                ), row=1, col=2)
    
    # Create aligned visualization (right plot)
    for word_type in ['theme', 'bible_similar', 'quran_similar']:
        mask = [word['type'] == word_type for word in word_data]
        if any(mask):
            x_coords = [coords[i][0] for i in range(len(coords)) if mask[i]]
            y_coords = [coords[i][1] for i in range(len(coords)) if mask[i]]
            z_coords = [coords[i][2] for i in range(len(coords)) if mask[i]]
            words = [word['word'] for word in word_data if word['type'] == word_type]
            similarities = [word.get('similarity', 1.0) for word in word_data if word['type'] == word_type]
            sources = [word['source'] for word in word_data if word['type'] == word_type]
            themes = [word.get('theme', 'unknown') for word in word_data if word['type'] == word_type]
            
            # Create hover text
            hover_text = []
            for word, sim, source, theme in zip(words, similarities, sources, themes):
                if word_type == 'theme':
                    hover_text.append(f"Word: {word}<br>Type: Theme Word<br>Source: {source}<br>Theme: {theme}<br>Similarity: {sim:.3f}")
                else:
                    hover_text.append(f"Word: {word}<br>Type: Similar Word<br>Source: {source}<br>Theme: {theme}<br>Similarity: {sim:.3f}")
            
            # Adjust point size based on similarity
            marker_sizes = [20 + sim * 30 for sim in similarities]
            
            # Use theme colors for theme words, source colors for similar words
            marker_colors = []
            for theme in themes:
                if word_type == 'theme':
                    marker_colors.append(theme_colors.get(theme, 'gold'))
                else:
                    marker_colors.append(source_colors[word_type])
            
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers+text',
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    opacity=0.8
                ),
                text=words,
                textposition="middle center",
                hovertext=hover_text,
                hoverinfo='text',
                name=f"{word_type.replace('_', ' ').title()}",
                showlegend=True
            ), row=1, col=3)
    
    # Update layout for all subplots
    fig.update_layout(
        title={
            'text': 'Bible and Quran Words Three-Dimensional Visualization',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        width=1800,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Update each subplot's scene
    for i in range(1, 4):
        fig.update_scenes(
            {
                'xaxis_title': 'X Axis',
                'yaxis_title': 'Y Axis',
                'zaxis_title': 'Z Axis',
                'camera': dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            },
            row=1, col=i
        )
    
    # Save basic HTML file
    output_file = 'enhanced_forgiveness_analysis.html'
    fig.write_html(output_file, include_plotlyjs=True)
    print(f"Basic visualization saved as: {output_file}")
    
    # Generate enhanced HTML with beautiful styling
    html_str = fig.to_html(include_plotlyjs=True, full_html=False)
    
    # Statistics
    bible_count = sum(1 for label in labels if label.startswith('Bible:'))
    quran_count = sum(1 for label in labels if label.startswith('Quran:'))
    theme_count = sum(1 for label in labels if 'forgiveness' in label)
    
    # Create enhanced HTML
    enhanced_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bible and Quran Words Three-Dimensional Visualization</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(15px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            h1 {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }}
            .info {{
                background: rgba(0, 0, 0, 0.4);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                border-left: 4px solid #4CAF50;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 25px;
                flex-wrap: wrap;
                gap: 15px;
            }}
            .stat-item {{
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                min-width: 120px;
            }}
            .stat-number {{
                font-size: 1.5em;
                font-weight: bold;
                color: #4CAF50;
            }}
            .legend {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-bottom: 25px;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                font-weight: bold;
            }}
            .legend-color {{
                width: 25px;
                height: 25px;
                border-radius: 50%;
                border: 2px solid white;
            }}
            .controls {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .controls button {{
                margin: 0 5px;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                background: #4CAF50;
                color: white;
                cursor: pointer;
                font-size: 14px;
            }}
            .controls button:hover {{
                background: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Bible and Quran Words Three-Dimensional Visualization</h1>
            
            <div class="info">
                <h3>🎯 Visualization Guide:</h3>
                <ul>
                    <li><strong>Left Plot:</strong> Bible words only - shows semantic structure of Bible vocabulary</li>
                    <li><strong>Middle Plot:</strong> Quran words only - shows semantic structure of Quran vocabulary</li>
                    <li><strong>Right Plot:</strong> Aligned vector spaces - compares both texts in the same coordinate system</li>
                </ul>
                <p><strong>🔬 Methodology:</strong> Separate Word2Vec models trained on Bible and Quran texts, then aligned using Procrustes analysis with MDS similarity preservation</p>
                <p><strong>🖱️ Interactive Controls:</strong> Mouse drag to rotate view, scroll to zoom, double-click to reset view</p>
                <p><strong>📊 Point Size:</strong> Larger points indicate higher similarity to theme words</p>
                <p><strong>🎨 Colors:</strong> Green = Theme words, Blue = Bible words, Red = Quran words</p>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{theme_count}</div>
                    <div>Theme Words</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{bible_count}</div>
                    <div>Bible Words</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{quran_count}</div>
                    <div>Quran Words</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(word_data)}</div>
                    <div>Total Words</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">0.5</div>
                    <div>Similarity Threshold</div>
                </div>
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: green;"></div>
                    <span>Theme Words</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: blue;"></div>
                    <span>Bible Words</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: red;"></div>
                    <span>Quran Words</span>
                </div>
            </div>
            
            <div class="controls">
                <button onclick="resetAllViews()">Reset All Views</button>
                <button onclick="exportData()">Export Data</button>
            </div>
            
            {html_str}
        </div>
        
        <script>
            function resetAllViews() {{
                // Reset all 3D plots' views
                const plotDivs = document.querySelectorAll('.plotly-graph-div');
                plotDivs.forEach((plotDiv, index) => {{
                    if (plotDiv && plotDiv._fullData) {{
                        Plotly.relayout(plotDiv, {{
                            'scene.camera': {{
                                'eye': {{'x': 1.5, 'y': 1.5, 'z': 1.5}}
                            }}
                        }});
                    }}
                }});
            }}
            
            function exportData() {{
                // Export data functionality
                const data = {{
                    word_data: {json.dumps(word_data, ensure_ascii=False, indent=2)},
                    coords: {json.dumps(coords, indent=2)},
                    labels: {json.dumps(labels, ensure_ascii=False, indent=2)},
                    colors: {json.dumps(colors, indent=2)}
                }};
                
                const dataStr = JSON.stringify(data, null, 2);
                const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'enhanced_forgiveness_analysis_data.json';
                link.click();
                URL.revokeObjectURL(url);
            }}
        </script>
    </body>
    </html>
    """
    
    # Save enhanced HTML file
    enhanced_output_file = 'enhanced_forgiveness_analysis_beautiful.html'
    with open(enhanced_output_file, 'w', encoding='utf-8') as f:
        f.write(enhanced_html)
    
    print(f"Beautiful enhanced visualization saved as: {enhanced_output_file}")
    
    return fig

if __name__ == '__main__':
    print("🔄 Starting enhanced forgiveness theme analysis...")
    
    # Create enhanced visualization (generates both basic and beautiful versions)
    fig = create_enhanced_visualization()
    
    print("\n✅ Enhanced analysis completed!")
    print("📊 Generated two HTML files:")
    print("   - enhanced_forgiveness_analysis.html (Basic version)")
    print("   - enhanced_forgiveness_analysis_beautiful.html (Beautiful enhanced version)")
    print("\n🎯 This version uses multiple forgiveness-related keywords:")
    print("   - forgiveness, forgive, forgiven, forgiving")
    print("   - mercy, merciful, compassion, compassionate") 
    print("   - pardon, pardoned, absolution, remission")
    print("   - redemption, redeem, redeemed, salvation")
    print("   - grace, gracious, favor, blessing")
    print("\n🔬 Features:")
    print("   - MDS similarity preservation")
    print("   - Procrustes alignment")
    print("   - Interactive 3D visualization")
    print("   - Beautiful HTML styling with statistics")
    print("\nYou can open these HTML files directly in your browser to view the interactive 3D visualizations") 
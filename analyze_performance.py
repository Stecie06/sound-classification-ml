"""
Analyze load test results and compare container scaling performance
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from datetime import datetime

def analyze_load_tests():
    """Analyze all load test results"""
    
    # Find all load test result files
    result_files = glob.glob("load_test_results_*_requests.csv")
    
    if not result_files:
        print("No load test result files found!")
        return
    
    results = []
    
    for file in result_files:
        # Extract container and user info from filename
        parts = file.split('_')
        containers = int(parts[3])
        users = int(parts[5])
        
        try:
            df = pd.read_csv(file)
            
            # Calculate metrics
            avg_response_time = df['Average Response Time'].mean()
            max_response_time = df['Average Response Time'].max()
            min_response_time = df['Average Response Time'].min()
            requests_per_second = len(df) / (df['Total Time'] / 1000).mean() if len(df) > 0 else 0
            
            results.append({
                'containers': containers,
                'users': users,
                'avg_response_time_ms': avg_response_time,
                'max_response_time_ms': max_response_time,
                'min_response_time_ms': min_response_time,
                'requests_per_second': requests_per_second,
                'total_requests': len(df),
                'test_file': file
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No valid results to analyze!")
        return
    
    print("📊 LOAD TEST PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    for _, row in results_df.iterrows():
        print(f"\nContainers: {row['containers']}, Users: {row['users']}")
        print(f"  Avg Response Time: {row['avg_response_time_ms']:.2f} ms")
        print(f"  Min/Max Response: {row['min_response_time_ms']:.2f} / {row['max_response_time_ms']:.2f} ms")
        print(f"  Requests/sec: {row['requests_per_second']:.2f}")
        print(f"  Total Requests: {row['total_requests']}")
    
    # Create visualization
    create_performance_charts(results_df)

def create_performance_charts(results_df):
    """Create performance comparison charts"""
    
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Chart 1: Response Time vs Containers
    for users in results_df['users'].unique():
        user_data = results_df[results_df['users'] == users]
        ax1.plot(user_data['containers'], user_data['avg_response_time_ms'], 
                marker='o', label=f'{users} Users')
    
    ax1.set_xlabel('Number of Containers')
    ax1.set_ylabel('Average Response Time (ms)')
    ax1.set_title('Response Time vs Container Scaling')
    ax1.legend()
    ax1.grid(True)
    
    # Chart 2: Throughput vs Containers
    for users in results_df['users'].unique():
        user_data = results_df[results_df['users'] == users]
        ax2.plot(user_data['containers'], user_data['requests_per_second'], 
                marker='s', label=f'{users} Users')
    
    ax2.set_xlabel('Number of Containers')
    ax2.set_ylabel('Requests per Second')
    ax2.set_title('Throughput vs Container Scaling')
    ax2.legend()
    ax2.grid(True)
    
    # Chart 3: Response Time Distribution
    container_groups = results_df.groupby('containers')
    response_times_by_container = []
    container_labels = []
    
    for containers, group in container_groups:
        response_times_by_container.append(group['avg_response_time_ms'].values)
        container_labels.append(f'{containers} Containers')
    
    ax3.boxplot(response_times_by_container, labels=container_labels)
    ax3.set_ylabel('Response Time (ms)')
    ax3.set_title('Response Time Distribution by Container Count')
    ax3.grid(True)
    
    # Chart 4: Performance Improvement
    baseline = results_df[results_df['containers'] == 1]['avg_response_time_ms'].values
    if len(baseline) > 0:
        baseline_time = baseline[0]
        improvement = []
        containers_list = sorted(results_df['containers'].unique())
        
        for containers in containers_list:
            if containers > 1:
                current_time = results_df[results_df['containers'] == containers]['avg_response_time_ms'].mean()
                improvement.append((baseline_time - current_time) / baseline_time * 100)
            else:
                improvement.append(0)
        
        ax4.bar(containers_list, improvement, color='green', alpha=0.7)
        ax4.set_xlabel('Number of Containers')
        ax4.set_ylabel('Performance Improvement (%)')
        ax4.set_title('Performance Improvement from Scaling')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    results_df.to_csv('performance_analysis_results.csv', index=False)
    print(f"\n✅ Analysis complete! Results saved to:")
    print("   - performance_analysis.png")
    print("   - performance_analysis_results.csv")

if __name__ == "__main__":
    analyze_load_tests()
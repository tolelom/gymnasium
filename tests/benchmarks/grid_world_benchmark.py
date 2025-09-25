"""GridWorld 환경 성능 벤치마크"""

import time
import numpy as np
import gymnasium as gym
import gymnasium_env
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
import psutil
import os

try:
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스"""
    test_name: str
    duration: float
    fps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    additional_metrics: Dict[str, Any]


class GridWorldBenchmark:
    """GridWorld 환경 벤치마크 클래스"""

    def __init__(self, sizes: List[int] = [5, 10, 20]):
        """
        Args:
            sizes: 테스트할 그리드 크기 리스트
        """
        self.sizes = sizes
        self.results: List[BenchmarkResult] = []

    def benchmark_environment_creation(self) -> None:
        """환경 생성 속도 벤치마크"""
        print("🏗️  Environment Creation Benchmark")

        for size in self.sizes:
            n_envs = 100
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # 환경 생성 및 해제
            for _ in range(n_envs):
                env = gym.make('gymnasium_env/GridWorld-v0', size=size)
                env.close()

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            duration = end_time - start_time
            envs_per_second = n_envs / duration
            memory_diff = end_memory - start_memory

            result = BenchmarkResult(
                test_name=f"env_creation_size_{size}",
                duration=duration,
                fps=envs_per_second,
                memory_usage_mb=memory_diff,
                cpu_usage_percent=psutil.cpu_percent(),
                additional_metrics={
                    'grid_size': size,
                    'num_envs': n_envs,
                    'envs_per_second': envs_per_second
                }
            )

            self.results.append(result)
            print(f"  Size {size}x{size}: {envs_per_second:.1f} envs/sec, "
                  f"{duration:.3f}s total, {memory_diff:.1f}MB")

    def benchmark_step_performance(self) -> None:
        """스텝 실행 성능 벤치마크"""
        print("\n⚡ Step Performance Benchmark")

        for size in self.sizes:
            env = gym.make('gymnasium_env/GridWorld-v0', size=size)

            n_steps = 10000
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_start = psutil.cpu_percent()

            obs, _ = env.reset()
            for _ in range(n_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_end = psutil.cpu_percent()

            duration = end_time - start_time
            steps_per_second = n_steps / duration
            memory_diff = end_memory - start_memory
            cpu_usage = (cpu_start + cpu_end) / 2

            result = BenchmarkResult(
                test_name=f"step_performance_size_{size}",
                duration=duration,
                fps=steps_per_second,
                memory_usage_mb=memory_diff,
                cpu_usage_percent=cpu_usage,
                additional_metrics={
                    'grid_size': size,
                    'num_steps': n_steps,
                    'steps_per_second': steps_per_second
                }
            )

            self.results.append(result)
            print(f"  Size {size}x{size}: {steps_per_second:.0f} steps/sec, "
                  f"CPU: {cpu_usage:.1f}%, Memory: +{memory_diff:.1f}MB")

            env.close()

    def benchmark_rendering_performance(self) -> None:
        """렌더링 성능 벤치마크"""
        print("\n🎨 Rendering Performance Benchmark")

        render_modes = ['human', 'rgb_array']

        for render_mode in render_modes:
            for size in self.sizes:
                env = gym.make('gymnasium_env/GridWorld-v0',
                               size=size, render_mode=render_mode)

                n_renders = 100 if render_mode == 'rgb_array' else 50
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024

                obs, _ = env.reset()
                for i in range(n_renders):
                    # 가끔 액션 실행해서 상태 변경
                    if i % 10 == 0:
                        action = env.action_space.sample()
                        obs, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            obs, _ = env.reset()

                    rendered = env.render()

                    # rgb_array 모드인 경우 배열 크기 확인
                    if render_mode == 'rgb_array' and rendered is not None:
                        assert isinstance(rendered, np.ndarray)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                duration = end_time - start_time
                renders_per_second = n_renders / duration
                memory_diff = end_memory - start_memory

                result = BenchmarkResult(
                    test_name=f"render_{render_mode}_size_{size}",
                    duration=duration,
                    fps=renders_per_second,
                    memory_usage_mb=memory_diff,
                    cpu_usage_percent=psutil.cpu_percent(),
                    additional_metrics={
                        'render_mode': render_mode,
                        'grid_size': size,
                        'num_renders': n_renders,
                        'renders_per_second': renders_per_second
                    }
                )

                self.results.append(result)
                print(f"  {render_mode} {size}x{size}: {renders_per_second:.1f} fps, "
                      f"Memory: +{memory_diff:.1f}MB")

                env.close()

    def benchmark_memory_usage(self) -> None:
        """메모리 사용량 벤치마크"""
        print("\n💾 Memory Usage Benchmark")

        for size in self.sizes:
            # 단일 환경 메모리 사용량
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            env = gym.make('gymnasium_env/GridWorld-v0', size=size)
            single_env_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # 다중 환경 메모리 사용량
            envs = []
            for _ in range(10):
                envs.append(gym.make('gymnasium_env/GridWorld-v0', size=size))

            multi_env_memory = psutil.Process().memory_info().rss / 1024 / 1024

            single_env_usage = single_env_memory - initial_memory
            multi_env_usage = multi_env_memory - initial_memory
            per_env_usage = (multi_env_memory - single_env_memory) / 9

            result = BenchmarkResult(
                test_name=f"memory_usage_size_{size}",
                duration=0.0,
                fps=0.0,
                memory_usage_mb=single_env_usage,
                cpu_usage_percent=0.0,
                additional_metrics={
                    'grid_size': size,
                    'single_env_mb': single_env_usage,
                    'multi_env_mb': multi_env_usage,
                    'per_env_mb': per_env_usage,
                    'num_multi_envs': 10
                }
            )

            self.results.append(result)
            print(f"  Size {size}x{size}: {single_env_usage:.1f}MB (single), "
                  f"{per_env_usage:.1f}MB (per additional env)")

            # 정리
            env.close()
            for e in envs:
                e.close()

    @staticmethod
    def benchmark_algorithm_training():
        """알고리즘 학습 성능 벤치마크"""
        if not SB3_AVAILABLE:
            print("\n⚠️  Stable Baselines3 not available, skipping training benchmark")
            return

        print("\n🤖 Algorithm Training Benchmark")

        algorithms = ['A2C', 'PPO']
        env_size = 5
        timesteps = 5000

        for algo_name in algorithms:
            env = gym.make('gymnasium_env/GridWorld-v0', size=env_size)

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            if algo_name == 'A2C':
                model = A2C("MultiInputPolicy", env, verbose=0)
            elif algo_name == 'PPO':
                model = PPO("MultiInputPolicy", env, verbose=0)

            model.learn(total_timesteps=timesteps)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            duration = end_time - start_time
            steps_per_second = timesteps / duration
            memory_diff = end_memory - start_memory

            print(f"  {algo_name}: {steps_per_second:.0f} steps/sec, "
                  f"{duration:.1f}s total, Memory: +{memory_diff:.1f}MB")

            env.close()

    def run_all_benchmarks(self) -> None:
        """모든 벤치마크 실행"""
        print("🚀 Starting GridWorld Benchmark Suite")
        print(f"Test sizes: {self.sizes}")
        print(f"System: {psutil.cpu_count()} CPUs, "
              f"{psutil.virtual_memory().total // 1024 // 1024 // 1024}GB RAM")
        print("-" * 60)

        self.benchmark_environment_creation()
        self.benchmark_step_performance()
        self.benchmark_rendering_performance()
        self.benchmark_memory_usage()
        self.benchmark_algorithm_training()

        print("\n✅ Benchmark Suite Completed")
        self.print_summary()

    def print_summary(self) -> None:
        """벤치마크 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY".center(80))
        print("=" * 80)

        # 카테고리별 결과 정리
        categories = {}
        for result in self.results:
            category = result.test_name.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        for category, results in categories.items():
            print(f"\n{category.upper()} RESULTS:")
            print("-" * 40)

            for result in results:
                print(f"  {result.test_name}: "
                      f"FPS={result.fps:.1f}, "
                      f"Memory={result.memory_usage_mb:.1f}MB, "
                      f"Duration={result.duration:.3f}s")

        print("\n" + "=" * 80)

    def plot_results(self, save_path: str = "benchmark_results.png") -> None:
        """벤치마크 결과 시각화"""
        if not self.results:
            print("No benchmark results to plot")
            return

        # 성능 결과만 추출
        step_results = [r for r in self.results if 'step_performance' in r.test_name]

        if not step_results:
            print("No step performance results to plot")
            return

        sizes = [r.additional_metrics['grid_size'] for r in step_results]
        fps_values = [r.fps for r in step_results]
        memory_values = [r.memory_usage_mb for r in step_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # FPS 그래프
        ax1.bar(sizes, fps_values)
        ax1.set_xlabel('Grid Size')
        ax1.set_ylabel('Steps per Second')
        ax1.set_title('Step Performance vs Grid Size')
        ax1.set_yscale('log')

        # 메모리 사용량 그래프
        ax2.bar(sizes, memory_values)
        ax2.set_xlabel('Grid Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Grid Size')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark plot saved to {save_path}")
        plt.show()

    def export_results(self, filepath: str = "benchmark_results.json") -> None:
        """벤치마크 결과를 JSON으로 내보내기"""
        import json
        from datetime import datetime

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total // 1024 // 1024 // 1024,
                'python_version': os.sys.version
            },
            'test_sizes': self.sizes,
            'results': []
        }

        for result in self.results:
            export_data['results'].append({
                'test_name': result.test_name,
                'duration': result.duration,
                'fps': result.fps,
                'memory_usage_mb': result.memory_usage_mb,
                'cpu_usage_percent': result.cpu_usage_percent,
                'additional_metrics': result.additional_metrics
            })

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Benchmark results exported to {filepath}")


def quick_benchmark():
    """빠른 벤치마크 실행"""
    benchmark = GridWorldBenchmark(sizes=[5, 10])
    benchmark.benchmark_step_performance()
    benchmark.print_summary()


def full_benchmark():
    """전체 벤치마크 실행"""
    benchmark = GridWorldBenchmark(sizes=[5, 10, 20])
    benchmark.run_all_benchmarks()
    benchmark.plot_results()
    benchmark.export_results()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_benchmark()
    else:
        full_benchmark()

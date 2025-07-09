# -*- coding: utf-8 -*-
"""
命令行工具主入口

提供各种命令行功能
"""

import click
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plate_recognition.pipeline.recognition_pipeline import RecognitionPipeline
from plate_recognition.core.config import get_config


@click.group()
@click.version_option(version='2.0.0')
def cli():
    """车牌识别系统命令行工具"""
    pass


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--method', default='comprehensive', 
              help='检测方法 (contour/color/comprehensive)')
@click.option('--engines', multiple=True, 
              help='OCR引擎 (tesseract/paddle/gemini)')
@click.option('--debug', is_flag=True, help='显示调试信息')
def recognize(image_path, method, engines, debug):
    """识别单张图片中的车牌"""
    try:
        pipeline = RecognitionPipeline()
        
        result = pipeline.recognize_from_file(
            image_path=image_path,
            detection_method=method,
            ocr_engines=list(engines) if engines else None,
            return_debug_info=debug
        )
        
        if result['success']:
            click.echo(f"✅ 识别成功: {result['plate_number']}")
            click.echo(f"置信度: {result['confidence']:.2f}")
            click.echo(f"处理时间: {result['processing_time']:.3f}s")
            
            if debug and 'debug_info' in result:
                click.echo("\n=== 调试信息 ===")
                debug_info = result['debug_info']
                for stage, info in debug_info.items():
                    click.echo(f"{stage}: {info}")
        else:
            click.echo(f"❌ 识别失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        click.echo(f"❌ 错误: {e}")
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='输出结果文件')
@click.option('--method', default='comprehensive', 
              help='检测方法 (contour/color/comprehensive)')
@click.option('--engines', multiple=True, 
              help='OCR引擎 (tesseract/paddle/gemini)')
def batch(input_dir, output, method, engines):
    """批量处理文件夹中的图片"""
    try:
        import json
        from datetime import datetime
        
        input_path = Path(input_dir)
        image_files = []
        
        # 查找图片文件
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        
        if not image_files:
            click.echo("❌ 没有找到图片文件")
            return
        
        click.echo(f"找到 {len(image_files)} 张图片")
        
        pipeline = RecognitionPipeline()
        results = []
        
        with click.progressbar(image_files, label='处理中') as files:
            for image_file in files:
                try:
                    result = pipeline.recognize_from_file(
                        image_path=image_file,
                        detection_method=method,
                        ocr_engines=list(engines) if engines else None
                    )
                    results.append(result)
                except Exception as e:
                    results.append({
                        'success': False,
                        'source_file': str(image_file),
                        'error': str(e)
                    })
        
        # 统计结果
        success_count = sum(1 for r in results if r.get('success'))
        success_rate = success_count / len(results) * 100
        
        click.echo(f"\n=== 处理完成 ===")
        click.echo(f"总数: {len(results)}")
        click.echo(f"成功: {success_count}")
        click.echo(f"失败: {len(results) - success_count}")
        click.echo(f"成功率: {success_rate:.1f}%")
        
        # 保存结果
        if output:
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': len(results),
                    'success': success_count,
                    'failed': len(results) - success_count,
                    'success_rate': success_rate
                },
                'results': results
            }
            
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            click.echo(f"结果已保存到: {output}")
        
    except Exception as e:
        click.echo(f"❌ 错误: {e}")
        sys.exit(1)


@cli.command()
def test():
    """运行系统测试"""
    try:
        from plate_recognition.pipeline.recognition_pipeline import RecognitionPipeline
        
        click.echo("🔧 初始化系统...")
        pipeline = RecognitionPipeline()
        
        click.echo("📊 系统状态:")
        stats = pipeline.get_stats()
        
        for component, status in stats['component_status'].items():
            status_icon = "✅" if status else "❌"
            click.echo(f"  {status_icon} {component}: {'可用' if status else '不可用'}")
        
        if stats['component_status']['ocr_engine']:
            engines = stats['component_status']['available_ocr_engines']
            click.echo(f"  📝 可用OCR引擎: {', '.join(engines)}")
        
        click.echo("\n✨ 系统测试完成")
        
    except Exception as e:
        click.echo(f"❌ 测试失败: {e}")
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='服务器地址')
@click.option('--port', default=5000, help='端口号')
@click.option('--debug', is_flag=True, help='调试模式')
def serve(host, port, debug):
    """启动API服务器"""
    try:
        from api.app import create_api_app
        
        app = create_api_app()
        
        click.echo(f"🚀 启动API服务器...")
        click.echo(f"地址: http://{host}:{port}")
        click.echo(f"健康检查: http://{host}:{port}/api/v1/health")
        click.echo(f"调试模式: {'开启' if debug else '关闭'}")
        
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        click.echo(f"❌ 服务器启动失败: {e}")
        sys.exit(1)


@cli.command()
def info():
    """显示系统信息"""
    try:
        config = get_config()
        
        click.echo("🔍 车牌识别系统信息")
        click.echo("=" * 40)
        click.echo(f"版本: 2.0.0")
        click.echo(f"环境: {config.environment}")
        click.echo(f"调试模式: {'开启' if config.debug else '关闭'}")
        click.echo(f"项目根目录: {config.project_root}")
        
        click.echo("\n📁 路径配置:")
        click.echo(f"模型路径: {config.get_model_path('')}")
        click.echo(f"日志路径: {config.get_log_path()}")
        click.echo(f"上传路径: {config.get_upload_path()}")
        
        click.echo("\n⚙️ OCR配置:")
        click.echo(f"Tesseract路径: {config.ocr.tesseract_cmd}")
        click.echo(f"PaddleOCR语言: {config.ocr.paddle_lang}")
        click.echo(f"使用GPU: {'是' if config.ocr.paddle_use_gpu else '否'}")
        
        click.echo("\n🌐 API配置:")
        click.echo(f"服务器地址: {config.api.host}:{config.api.port}")
        click.echo(f"最大文件大小: {config.api.max_content_length // (1024*1024)}MB")
        
    except Exception as e:
        click.echo(f"❌ 获取系统信息失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()

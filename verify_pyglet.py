import pyglet
import sys
import os

print("开始验证 Pyglet Headless 模式...")

# 强制开启 headless 模式。这是告诉 pyglet 不要寻找物理显示器的最直接方式。
pyglet.options['headless'] = True

# 定义要保存的图片文件名
output_filename = "pyglet_headless_test.png"

try:
    # 1. 创建一个 400x300 的不可见窗口
    window = pyglet.window.Window(width=400, height=300, visible=False)
    print("✅ 成功创建不可见窗口。")

    # 2. 创建一个要绘制的标签
    # 注意：服务器可能没有 Arial 字体。如果图片中没有文字，可以尝试换成 'DejaVu Sans' 或 'FreeSans'
    label = pyglet.text.Label('Pyglet Headless OK!',
                              font_name='DejaVu Sans',
                              font_size=24,
                              x=window.width // 2, y=window.height // 2,
                              anchor_x='center', anchor_y='center')
    print("✅ 成功创建文本标签。")

    # 3. 核心部分：手动调度一次绘制，保存，然后退出
    window.switch_to()
    window.dispatch_event('on_draw') # 手动触发 on_draw
    window.flip() # 将后台缓冲区的内容交换到前台（在 headless 模式下也需要）

    # 4. 从缓冲区抓取图像并保存
    pyglet.image.get_buffer_manager().get_color_buffer().save(output_filename)
    print(f"✅ 成功将缓冲区内容保存为 '{output_filename}'。")

    # 5. 清理并关闭
    window.close()
    print("✅ 窗口已关闭。")

except Exception as e:
    print(f"\n❌ 在验证过程中发生错误: {e}")
    raise e
    print("❌ Headless 环境配置可能存在问题。")
    sys.exit(1)

print("\n🎉 恭喜！Pyglet Headless 验证脚本顺利完成。")
print(f"请检查当前目录下是否生成了 '{output_filename}' 文件，并查看其内容是否正确。")
sys.exit(0)
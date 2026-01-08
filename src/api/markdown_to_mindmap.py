"""
将 markdown 文件转换为 React Flow 思维导图格式
"""
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def extract_heading_number(content: str) -> Optional[str]:
    """
    从标题内容中提取序号
    例如：
    - "## 3.1 点云采样模块" -> "3.1"
    - "# 4 实验" -> "4"
    - "### 3.1.1 细节" -> "3.1.1"
    """
    # 移除标题标记
    text = content.lstrip('#').strip()
    # 匹配开头的数字序号
    match = re.match(r'^(\d+(?:\.\d+)*)\s+', text)
    if match:
        return match.group(1)
    return None


def count_heading_hashes(content: str) -> int:
    """计算标题的 # 数量"""
    count = 0
    for char in content:
        if char == '#':
            count += 1
        else:
            break
    return count


def extract_block_text(block: Dict[str, Any]) -> str:
    """
    从块中提取文本内容（从 lines/spans 结构中）
    """
    lines = block.get('lines', [])
    if not lines:
        return ''
    
    text_parts = []
    for line in lines:
        spans = line.get('spans', [])
        for span in spans:
            content = span.get('content', '')
            if content:
                text_parts.append(content)
    
    return ' '.join(text_parts)


def find_bbox_in_middle_json(
    middle_json_path: Path,
    heading_level: int,
    number: str,
    get_next: bool = False
) -> List[float]:
    """
    在 middle_translate.json 中查找对应的 bbox
    
    Args:
        middle_json_path: middle_translate.json 文件路径
        heading_level: 标题级别（# 的数量）- 暂时不用于匹配
        number: 序号（如 "3.1"）
        get_next: 是否获取下一个块的 bbox（用于父节点匹配）
    
    Returns:
        bbox 坐标 [x, y, w, h]，失败返回 [0, 0, 0, 0]
    """
    try:
        with open(middle_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pdf_info = data.get('pdf_info', [])
        
        # 遍历所有页面
        for page in pdf_info:
            para_blocks = page.get('para_blocks', [])
            
            for i, block in enumerate(para_blocks):
                # 优先查找 type 为 title 的块
                if block.get('type') != 'title':
                    continue
                
                # 尝试从 translate_content 获取内容
                content = block.get('translate_content', '')
                
                # 如果 translate_content 为空，从 lines/spans 提取
                if not content:
                    content = extract_block_text(block)
                
                if not content:
                    continue
                
                # 检查序号（关键：只根据序号匹配，不检查标题级别）
                # 因为原文可能没有 # 标记
                block_number = extract_heading_number(content)
                if block_number != number:
                    continue
                
                # 找到匹配的块
                if get_next:
                    # 获取下一个块的 bbox
                    if i + 1 < len(para_blocks):
                        next_block = para_blocks[i + 1]
                        bbox = next_block.get('bbox', [0, 0, 0, 0])
                        return bbox
                else:
                    # 获取当前块的 bbox
                    bbox = block.get('bbox', [0, 0, 0, 0])
                    return bbox
        
        # 未找到匹配
        return [0, 0, 0, 0]
    
    except Exception as e:
        print(f"查找 bbox 失败: {e}")
        return [0, 0, 0, 0]


class MarkdownToMindmap:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_id_counter = 0
        
    def generate_node_id(self) -> str:
        """生成唯一的节点 ID"""
        if self.node_id_counter == 0:
            node_id = "1"
        else:
            timestamp = int(time.time() * 1000)
            node_id = f"node_{timestamp}_{self.node_id_counter}"
        self.node_id_counter += 1
        return node_id
    
    def create_node(self, content: str, sort_index: int = 0, child_count: int = 0, bboxs: List[float] = None) -> Dict[str, Any]:
        """创建一个思维导图节点"""
        node_id = self.generate_node_id()
        if bboxs is None:
            bboxs = [0, 0, 0, 0]
        
        node = {
            "id": node_id,
            "data": {
                "sortIndex": sort_index,
                "childCount": child_count,
                "contentType": "text",
                "markdownContent": content,
                "bboxs": bboxs
            },
            "type": "mindmap"
        }
        return node
    
    def create_edge(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """创建节点之间的连接"""
        edge = {
            "id": f"e{source_id}-{target_id}",
            "source": source_id,
            "target": target_id
        }
        return edge
    
    def parse_heading_level(self, line: str) -> Tuple[int, str, bool]:
        """
        解析标题级别
        返回: (级别, 标题内容, 是否有序号)
        """
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if not match:
            return 0, "", False
        
        level = len(match.group(1))
        content = match.group(2).strip()
        
        # 检查是否有序号（如 "1 引言", "2.1 人体姿态估计"）
        has_number = bool(re.match(r'^\d+(\.\d+)*\s+', content))
        
        return level, content, has_number
    
    def extract_heading_number(self, content: str) -> Optional[str]:
        """提取标题序号"""
        match = re.match(r'^(\d+(\.\d+)*)\s+', content)
        if match:
            return match.group(1)
        return None
    
    def parse_markdown(self, md_content: str) -> List[Dict[str, Any]]:
        """
        解析 markdown 内容，返回结构化数据
        每个元素包含: type (heading/content), level, content, has_number, number
        """
        lines = md_content.split('\n')
        elements = []
        current_content = []
        
        for line in lines:
            # 检查是否是标题
            level, heading_content, has_number = self.parse_heading_level(line)
            
            if level > 0:
                # 如果有积累的正文内容，先保存
                if current_content:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        elements.append({
                            'type': 'content',
                            'content': content_text,
                            'level': 0
                        })
                    current_content = []
                
                # 添加标题
                number = self.extract_heading_number(heading_content) if has_number else None
                elements.append({
                    'type': 'heading',
                    'level': level,
                    'content': heading_content,
                    'has_number': has_number,
                    'number': number,
                    'original_line': line.strip()
                })
            else:
                # 累积正文内容
                if line.strip():  # 忽略空行
                    current_content.append(line)
        
        # 处理最后的正文内容
        if current_content:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                elements.append({
                    'type': 'content',
                    'content': content_text,
                    'level': 0
                })
        
        return elements
    
    def build_tree(self, elements: List[Dict[str, Any]], root_title: str) -> Dict[str, Any]:
        """
        构建树形结构
        """
        # 创建根节点
        root = {
            'type': 'heading',
            'level': 0,
            'content': root_title,
            'has_number': False,
            'number': None,
            'original_line': f"# {root_title}",
            'children': []
        }
        
        # 维护一个栈来跟踪当前路径
        # 栈中的元素格式: (节点, 标题级别, 是否有序号, 序号)
        stack = [(root, 0, False, None)]
        
        # 记录最近的有序号标题（用于无序号标题挂靠）
        last_numbered_by_level = {0: root}  # level -> node
        
        i = 0
        while i < len(elements):
            elem = elements[i]
            
            if elem['type'] == 'heading':
                current_level = elem['level']
                has_number = elem['has_number']
                number = elem.get('number')
                
                # 找到合适的父节点
                if has_number:
                    # 有序号的标题：根据序号层级挂靠
                    # 例如: 2.1 挂在 2 下面, 2.1.1 挂在 2.1 下面
                    if number:
                        parts = number.split('.')
                        depth = len(parts)
                        
                        # 找到对应深度的父节点
                        if depth == 1:
                            # 一级标题，挂在根节点下
                            parent = root
                        else:
                            # 多级标题，找到上一级
                            parent_number = '.'.join(parts[:-1])
                            # 在 last_numbered_by_level 中查找
                            parent = None
                            for level, node in sorted(last_numbered_by_level.items(), reverse=True):
                                if node.get('number') == parent_number:
                                    parent = node
                                    break
                            if parent is None:
                                # 如果找不到，挂在根节点下
                                parent = root
                        
                        # 更新 last_numbered_by_level
                        last_numbered_by_level[depth] = elem
                    else:
                        parent = root
                else:
                    # 无序号的标题：找到其上面最近的有序号标题
                    parent = None
                    for level in sorted(last_numbered_by_level.keys(), reverse=True):
                        if level > 0:  # 不使用根节点
                            parent = last_numbered_by_level[level]
                            break
                    if parent is None:
                        parent = root
                    
                    # 强制调整无序号标题的层级，使其符合父节点的子标题层级
                    # 父节点是 # (level=1)，则无序号标题应该是 ## (level=2)
                    parent_level = parent.get('level', 0)
                    expected_level = parent_level + 1
                    
                    if current_level != expected_level:
                        # 调整标题级别
                        old_level = current_level
                        elem['level'] = expected_level
                        
                        # 更新 original_line 以反映新的层级
                        heading_prefix = '#' * expected_level
                        # 移除原来的 # 前缀
                        content = elem['content']
                        elem['original_line'] = f"{heading_prefix} {content}"
                
                # 添加标题节点
                elem['children'] = []
                if 'children' not in parent:
                    parent['children'] = []
                parent['children'].append(elem)
                
                # 检查下一个元素是否是正文内容
                if i + 1 < len(elements) and elements[i + 1]['type'] == 'content':
                    content_elem = elements[i + 1]
                    elem['children'].append(content_elem)
                    i += 1  # 跳过正文内容
            
            i += 1
        
        return root
    
    def check_node_conditions(self, content: str) -> Tuple[bool, int, Optional[str]]:
        """
        检查节点是否满足条件（标题 + 序号）
        
        Returns:
            (是否满足条件, 标题级别, 序号)
        """
        first_line = content.split('\n')[0]
        
        # 检查是否是标题
        if not first_line.startswith('#'):
            return False, 0, None
        
        # 获取标题级别
        level = count_heading_hashes(first_line)
        
        # 提取序号
        number = extract_heading_number(first_line)
        
        # 必须同时有标题和序号
        if number is not None:
            return True, level, number
        
        return False, level, None
    
    def find_parent_with_conditions(self, tree: Dict[str, Any], node_map: Dict[str, Dict]) -> Optional[Tuple[int, str]]:
        """
        递归查找满足条件的父节点
        
        Returns:
            (标题级别, 序号) 或 None
        """
        # 从 node_map 中查找父节点
        for node_id, node_info in node_map.items():
            children = node_info.get('children', [])
            for child in children:
                if child.get('id') == tree.get('id'):
                    # 找到父节点，检查是否满足条件
                    parent_content = node_info.get('content', '')
                    satisfies, level, number = self.check_node_conditions(parent_content)
                    if satisfies:
                        return level, number
                    # 如果父节点不满足，继续往上找
                    return self.find_parent_with_conditions(node_info, node_map)
        
        return None
    
    def tree_to_mindmap(self, tree: Dict[str, Any], parent_id: Optional[str] = None, sort_index: int = 0, parent_tree: Dict[str, Any] = None):
        """
        将树形结构转换为 React Flow 格式的节点和边
        """
        # 创建当前节点的内容
        if tree['type'] == 'heading':
            content = tree['original_line']
        else:
            content = tree['content']
        
        # 查找 bbox
        bboxs = [0, 0, 0, 0]  # 默认值
        
        if hasattr(self, 'middle_json_path') and self.middle_json_path:
            # 检查当前节点是否满足条件
            satisfies, level, number = self.check_node_conditions(content)
            
            if satisfies:
                # 自身满足条件，直接查找 bbox
                bboxs = find_bbox_in_middle_json(
                    self.middle_json_path,
                    level,
                    number,
                    get_next=False
                )
            elif parent_tree is not None:
                # 自身不满足，查找父节点
                parent_content = parent_tree.get('original_line', '') if parent_tree.get('type') == 'heading' else parent_tree.get('content', '')
                parent_satisfies, parent_level, parent_number = self.check_node_conditions(parent_content)
                
                if parent_satisfies:
                    # 父节点满足条件，获取下一个块的 bbox
                    bboxs = find_bbox_in_middle_json(
                        self.middle_json_path,
                        parent_level,
                        parent_number,
                        get_next=True
                    )
                else:
                    # 父节点也不满足，继续往上找
                    # 这里需要递归查找，暂时保持默认值
                    pass
        
        # 创建节点
        child_count = len(tree.get('children', []))
        node = self.create_node(content, sort_index, child_count, bboxs)
        self.nodes.append(node)
        
        # 创建与父节点的连接
        if parent_id is not None:
            edge = self.create_edge(parent_id, node['id'])
            self.edges.append(edge)
        
        # 递归处理子节点，传递当前树作为父树
        children = tree.get('children', [])
        for idx, child in enumerate(children):
            self.tree_to_mindmap(child, node['id'], idx, tree)
    
    def convert(self, md_file_path: str, output_file_path: str, file_title: str, middle_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        转换 markdown 文件为思维导图 JSON
        
        Args:
            md_file_path: markdown 文件路径
            output_file_path: 输出 JSON 文件路径
            file_title: 根节点标题（通常是文件名）
            middle_json_path: middle_translate.json 文件路径（可选）
        
        Returns:
            思维导图 JSON 数据
        """
        # 保存 middle_json_path 供后续使用
        if middle_json_path:
            self.middle_json_path = Path(middle_json_path)
        else:
            self.middle_json_path = None
        
        # 读取 markdown 文件
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 解析 markdown
        elements = self.parse_markdown(md_content)
        
        # 构建树形结构
        tree = self.build_tree(elements, file_title)
        
        # 转换为思维导图格式
        self.tree_to_mindmap(tree)
        
        # 构建最终的 JSON 结构
        result = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        
        # 保存到文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result


def find_mindmap_explain_file(base_dir: str, file_name: str) -> Optional[str]:
    """
    在 output/mineru 目录下查找 *_mindmap_explain.md 文件
    
    Args:
        base_dir: 项目根目录
        file_name: 文件名（不含扩展名）
    
    Returns:
        找到的文件路径，如果没找到返回 None
    """
    output_dir = Path(base_dir) / "output" / "mineru" / file_name
    
    if not output_dir.exists():
        return None
    
    # 递归查找 *_mindmap_explain.md 文件
    for md_file in output_dir.rglob("*_mindmap_explain.md"):
        return str(md_file)
    
    return None


def convert_markdown_to_mindmap(file_name: str, base_dir: str = ".") -> Optional[Dict[str, Any]]:
    """
    将指定文件名的 markdown 转换为思维导图
    
    Args:
        file_name: 文件名（不含扩展名）
        base_dir: 项目根目录
    
    Returns:
        思维导图 JSON 数据，如果失败返回 None
    """
    # 查找文件
    md_file_path = find_mindmap_explain_file(base_dir, file_name)
    
    if md_file_path is None:
        print(f"未找到文件: {file_name}")
        return None
    
    print(f"找到文件: {md_file_path}")
    
    # 查找 middle_translate.json 文件（在同级目录）
    md_path = Path(md_file_path)
    middle_json_path = md_path.parent / f"{file_name}_middle_translate.json"
    
    if not middle_json_path.exists():
        # 尝试查找不带后缀的
        middle_json_path = md_path.parent / "middle_translate.json"
    
    if middle_json_path.exists():
        print(f"找到 middle_translate.json: {middle_json_path}")
    else:
        print(f"未找到 middle_translate.json，将使用默认 bbox")
        middle_json_path = None
    
    # 生成输出文件路径（与输入文件同级目录）
    output_file_path = md_path.parent / f"{md_path.stem}_mindmap.json"
    
    # 转换
    converter = MarkdownToMindmap()
    result = converter.convert(
        str(md_path),
        str(output_file_path),
        file_name,
        str(middle_json_path) if middle_json_path else None
    )
    
    print(f"思维导图已保存到: {output_file_path}")
    
    return result


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python markdown_to_mindmap.py <文件名>")
        sys.exit(1)
    
    file_name = sys.argv[1]
    result = convert_markdown_to_mindmap(file_name, "/Volumes/haoyu_2t/Code/Python/LinearRAG")
    
    if result:
        print(f"生成了 {len(result['nodes'])} 个节点和 {len(result['edges'])} 条边")


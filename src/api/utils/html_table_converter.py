"""HTML 表格转 Markdown 表格工具"""
import re
from html.parser import HTMLParser
from typing import List, Tuple, Optional


class HTMLTableParser(HTMLParser):
    """解析 HTML 表格为结构化数据"""
    
    def __init__(self):
        super().__init__()
        self.tables = []
        self.current_table = []
        self.current_row = []
        self.current_cell = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
            self.current_table = []
        elif tag == 'tr' and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ('td', 'th') and self.in_row:
            self.in_cell = True
            self.current_cell = []
            # 保存单元格属性（如 rowspan, colspan）
            self.cell_attrs = dict(attrs)
            
    def handle_endtag(self, tag):
        if tag == 'table' and self.in_table:
            self.in_table = False
            if self.current_table:
                self.tables.append(self.current_table)
                self.current_table = []
        elif tag == 'tr' and self.in_row:
            self.in_row = False
            if self.current_row:
                self.current_table.append(self.current_row)
                self.current_row = []
        elif tag in ('td', 'th') and self.in_cell:
            self.in_cell = False
            cell_text = ''.join(self.current_cell).strip()
            rowspan = int(self.cell_attrs.get('rowspan', 1))
            colspan = int(self.cell_attrs.get('colspan', 1))
            self.current_row.append({
                'text': cell_text,
                'rowspan': rowspan,
                'colspan': colspan,
                'is_header': tag == 'th'
            })
            self.current_cell = []
            self.cell_attrs = {}
            
    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def parse_html_tables(html_text: str) -> List[List[List[dict]]]:
    """解析 HTML 文本中的所有表格
    
    Args:
        html_text: 包含 HTML 表格的文本
        
    Returns:
        表格列表，每个表格是行的列表，每行是单元格字典的列表
    """
    parser = HTMLTableParser()
    parser.feed(html_text)
    return parser.tables


def html_table_to_markdown(table_data: List[List[dict]]) -> str:
    """将解析后的表格数据转换为 Markdown 表格
    
    Args:
        table_data: 表格数据（行的列表，每行是单元格字典的列表）
        
    Returns:
        Markdown 格式的表格字符串
    """
    if not table_data:
        return ""
    
    # 计算表格的总列数（考虑 colspan）
    max_cols = 0
    for row_data in table_data:
        cols = sum(cell.get('colspan', 1) for cell in row_data)
        max_cols = max(max_cols, cols)
    
    # 创建二维数组表示表格
    grid = []
    rowspan_tracker = {}  # 记录每列的 rowspan 状态 {col_idx: (remaining_rows, text)}
    
    for row_idx, row_data in enumerate(table_data):
        row = [''] * max_cols
        col_idx = 0
        
        # 首先填充由前面行的 rowspan 占据的单元格
        for c in range(max_cols):
            if c in rowspan_tracker and rowspan_tracker[c][0] > 0:
                row[c] = rowspan_tracker[c][1]
                # 减少剩余行数
                rowspan_tracker[c] = (rowspan_tracker[c][0] - 1, rowspan_tracker[c][1])
                if col_idx == c:
                    col_idx += 1
        
        # 填充当前行的数据
        for cell in row_data:
            # 跳过被 rowspan 占据的列
            while col_idx < max_cols and col_idx in rowspan_tracker and rowspan_tracker[col_idx][0] > 0:
                col_idx += 1
            
            if col_idx >= max_cols:
                break
            
            text = cell['text']
            colspan = cell.get('colspan', 1)
            rowspan = cell.get('rowspan', 1)
            
            # 填充当前单元格和 colspan 单元格
            for c in range(colspan):
                if col_idx + c < max_cols:
                    row[col_idx + c] = text if c == 0 else ''
            
            # 如果有 rowspan，记录到 tracker
            if rowspan > 1:
                for c in range(colspan):
                    if col_idx + c < max_cols:
                        rowspan_tracker[col_idx + c] = (rowspan - 1, text if c == 0 else '')
            
            col_idx += colspan
        
        grid.append(row)
    
    if not grid:
        return ""
    
    # 生成 Markdown 表格
    lines = []
    
    # 第一行（表头）
    if grid:
        header = grid[0]
        lines.append('| ' + ' | '.join(header) + ' |')
        # 分隔线
        lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
        
        # 数据行
        for row in grid[1:]:
            lines.append('| ' + ' | '.join(row) + ' |')
    
    return '\n'.join(lines)


def convert_html_tables_in_markdown(markdown_text: str) -> Tuple[str, int]:
    """将 Markdown 文本中的所有 HTML 表格转换为 Markdown 表格
    
    Args:
        markdown_text: 包含 HTML 表格的 Markdown 文本
        
    Returns:
        (转换后的文本, 转换的表格数量)
    """
    # 查找所有 <table>...</table> 标签
    table_pattern = re.compile(r'<table>.*?</table>', re.DOTALL | re.IGNORECASE)
    
    tables_found = table_pattern.findall(markdown_text)
    if not tables_found:
        return markdown_text, 0
    
    converted_count = 0
    result = markdown_text
    
    for html_table in tables_found:
        # 解析 HTML 表格
        parsed_tables = parse_html_tables(html_table)
        
        if parsed_tables:
            # 转换为 Markdown
            markdown_table = html_table_to_markdown(parsed_tables[0])
            
            if markdown_table:
                # 替换原始 HTML 表格
                result = result.replace(html_table, markdown_table, 1)
                converted_count += 1
    
    return result, converted_count


def preprocess_markdown_for_docx(markdown_text: str) -> Tuple[str, dict]:
    """预处理 Markdown 文本以便转换为 DOCX
    
    Args:
        markdown_text: 原始 Markdown 文本
        
    Returns:
        (处理后的文本, 处理统计信息)
    """
    stats = {
        'html_tables_converted': 0,
        'original_length': len(markdown_text),
    }
    
    # 转换 HTML 表格
    processed_text, table_count = convert_html_tables_in_markdown(markdown_text)
    stats['html_tables_converted'] = table_count
    stats['processed_length'] = len(processed_text)
    
    return processed_text, stats


"""ResponseLab 界面共享的小型控件。"""

from PySide6.QtWidgets import QDoubleSpinBox


class CompactDoubleSpinBox(QDoubleSpinBox):
    """保留输入精度，同时隐藏没有信息量的末尾零。"""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        text = super().textFromValue(value)
        decimal_point = self.locale().decimalPoint()
        if decimal_point in text:
            text = text.rstrip("0").rstrip(decimal_point)
        return text

from abc import ABC, abstractmethod
from typing import Dict, Optional

# متادیتای پیش‌فرض برای BaseAI
AI_METADATA = {
    "default_type": "base_ai",
    "default_description": "هوش مصنوعی پایه با قابلیت‌های اولیه برای بازی شطرنج."
}

class BaseAI(ABC):
    """کلاس پایه انتزاعی برای هوش‌های مصنوعی در بازی شطرنج."""

    def __init__(self, game, model_name: str, ai_id: str, settings: Optional[Dict] = None):
        """
        مقداردهی اولیه هوش مصنوعی پایه.

        Args:
            game: شیء بازی (CheckersGame)
            model_name: نام مدل برای ذخیره/بارگذاری
            ai_id: شناسه منحصربه‌فرد برای AI (مثل 'ai_1' یا 'ai_2')
            settings: تنظیمات اختیاری از config.json
        """
        self.game = game
        self.model_name = model_name
        self.ai_id = ai_id
        self.settings = settings or {}

    @abstractmethod
    def get_move(self, board):
        """
        انتخاب حرکت برای تخته فعلی.

        Args:
            board: وضعیت فعلی تخته بازی

        Returns:
            حرکت انتخاب‌شده (مثل یک tuple یا dict)
        """
        pass

    @abstractmethod
    def update(self, move, reward):
        """
        به‌روزرسانی مدل AI بر اساس حرکت و پاداش دریافتی.

        Args:
            move: حرکت انجام‌شده
            reward: پاداش دریافتی
        """
        pass

    @classmethod
    def get_metadata(cls) -> Dict[str, str]:
        """
        دریافت متادیتای پیش‌فرض برای این AI.

        Returns:
            دیکشنری شامل default_type و default_description
        """
        return getattr(cls, "AI_METADATA", AI_METADATA)
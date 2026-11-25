"""Catalog completeness and the stored language preference."""
import ast
import json
import string
import tempfile
import unittest
from pathlib import Path

import lab_plotter_i18n as i18n
from lab_plotter_config import SIGNAL_ROLES
from lab_plotter_core import FIGURE_TYPES, subplot_traces

APP_SOURCE = Path(__file__).with_name('lab_plotter_app.py')


def fields(text):
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


class TranslationTests(unittest.TestCase):
    def test_catalog_covers_interface_texts_and_keeps_placeholders(self):
        texts = set()
        for node in ast.walk(ast.parse(APP_SOURCE.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            argument = node.args[0] if name in ('tr', 'translate') else node.args[-1]
            if (name in ('tr', 'translate', '_t', '_heading', 'set_file_state', 'update_row')
                    and isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
                texts.add(argument.value)
        texts |= {label for _, label, *_ in SIGNAL_ROLES}
        texts |= {trace['title'] for kind in FIGURE_TYPES for trace in subplot_traces(kind)}
        self.assertGreater(len(texts), 100)
        self.assertEqual(sorted(text for text in texts if text not in i18n.FINNISH and text not in i18n.NEUTRAL), [])
        for english, finnish in i18n.FINNISH.items():
            with self.subTest(text=english):
                self.assertEqual(fields(english), fields(finnish))

    def test_translate_falls_back_to_english_and_fills_fields(self):
        self.assertEqual(i18n.translate('Preview: {path}', 'fi', path='a.mat'), 'Esikatselu: a.mat')
        self.assertEqual(i18n.translate('Preview: {path}', 'en', path='a.mat'), 'Preview: a.mat')
        self.assertEqual(i18n.translate('({role} not linked)', 'fi'), '({role} ei linkitetty)')
        self.assertEqual(i18n.translate('Unlisted text', 'fi'), 'Unlisted text')

    def test_language_preference_is_stored_per_user(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            self.assertEqual(i18n.load_language(path), 'fi')
            for text in ('{invalid', '[]', '{"language": "sv"}'):
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text):
                    self.assertEqual(i18n.load_language(path), 'fi')
            path.write_text(json.dumps({'other': 1}), encoding='utf-8')
            self.assertTrue(i18n.save_language('en', path))
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'other': 1, 'language': 'en'})
            self.assertEqual(i18n.load_language(path), 'en')
            with self.assertRaises(ValueError):
                i18n.save_language('sv', path)
            self.assertFalse(i18n.save_language('fi', Path(folder) / 'missing' / 'settings.json'))
            self.assertEqual(sorted(item.name for item in Path(folder).iterdir()), ['settings.json'])


if __name__ == '__main__':
    unittest.main()

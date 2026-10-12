"""Metric row isolation tests use source documents, never reference answers."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/rag'))
from table_semantics_v4 import metric_scoped_table
from select_context_v2 import Slot, table_binding_match


def table(rows, header=(1,), data=None, caption='财务指标', company='甲公司'):
    return {'content_type': 'table', 'company': company, 'table_caption': caption,
            'header_row_numbers': list(header), 'data_row_numbers': data or list(range(2, len(rows)+1)),
            'rows': [{'row': i+1, 'cells': list(c), 'display_cells': list(c), 'raw_cells': list(c)}
                     for i, c in enumerate(rows)],
            'text': '原文：净利润、营业收入和固定资产，单位万元', 'content_text': '原始财务表格文本',
            'source_path': 'original.tex'}


class TableSemanticsTests(unittest.TestCase):
    def test_real_split_income_profit_groups_preserve_negative_profit(self):
        if not (ROOT/"data/processed/corpus/evidence_chunks.jsonl").exists():
            self.skipTest("Optional real-corpus integration fixture is not distributed")
        chunks = [json.loads(s) for s in (ROOT/'data/processed/corpus/evidence_chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        original = next(c for c in chunks if c['chunk_id']=='百奥赛图-B_企业财务报告_2022年度财务分析::t0005::c001')
        scoped = metric_scoped_table(original, '净利润', '百奥赛图-B')
        self.assertEqual([r['row'] for r in scoped['rows']], [1,2,3,5])
        row = next(r for r in scoped['rows'] if r['row']==5)
        self.assertIn('百奥赛图-B', row['display_cells'][0])
        self.assertEqual(row['display_cells'][1], '-602.157')
        slot = Slot('net_profit', ('净利润',), '2022')
        self.assertTrue(table_binding_match(slot, scoped, '百奥赛图-B','百奥赛图-B','-602.157','百万元')['matched'])
        self.assertFalse(table_binding_match(slot, scoped, '百奥赛图-B','百奥赛图-B','533.881','百万元')['matched'])
        for key in original.keys()-{'rows'}:
            self.assertEqual(scoped[key], original[key])
        self.assertEqual(original['rows'][4]['display_cells'][0], '')

    def test_real_income_view_cannot_borrow_loss_row(self):
        if not (ROOT/"data/processed/corpus/evidence_chunks.jsonl").exists():
            self.skipTest("Optional real-corpus integration fixture is not distributed")
        original = next(json.loads(s) for s in (ROOT/'data/processed/corpus/evidence_chunks.jsonl').read_text(encoding='utf-8').splitlines()
                        if '百奥赛图-B_企业财务报告_2022年度财务分析::t0005::c001' in s)
        scoped = metric_scoped_table(original,'营业收入','百奥赛图-B')
        self.assertEqual([r['row'] for r in scoped['rows']], [1,2,3,4])

    def test_empty_and_prose_views_do_not_change_source(self):
        for c in ({}, {'content_text':'净利润', 'rows':[]}, {'rows':[{'row':1,'display_cells':[]}], 'header_row_numbers':[1]}):
            self.assertEqual(metric_scoped_table(c, '净利润','甲公司'), c)

    def test_longest_metric_does_not_merge_parent_profit_or_noncurrent_assets(self):
        original = table([['指标','2023','2024'],['净利润','10','12'],['归属于母公司股东的净利润','8','9'],
                          ['非流动资产合计','40','42'],['流动资产合计','20','25']])
        net = metric_scoped_table(original,'净利润','甲公司')
        self.assertEqual([r['row'] for r in net['rows']], [1,2])
        current = metric_scoped_table(original,'流动资产','甲公司')
        self.assertEqual([r['row'] for r in current['rows']], [1,5])

    def test_explicit_metric_rows_overrule_broad_caption(self):
        original = table([['指标','2023','2024'],['营业收入','100','120'],['营业成本','70','80']],caption='营业收入、营业成本、净利润')
        scoped = metric_scoped_table(original,'净利润','甲公司')
        self.assertEqual([r['row'] for r in scoped['rows']], [1])

    def test_annual_metric_row_and_unit_continuation_retained(self):
        original = table([['指标','2023','2024'],['营业收入','',''],['原始披露值','100','120'],['单位','万元','万元'],
                          ['营业成本','70','80'],['单位','万元','万元']])
        scoped = metric_scoped_table(original,'营业收入','甲公司')
        self.assertEqual([r['row'] for r in scoped['rows']], [1,2,3,4])
        self.assertEqual(scoped['rows'][2]['display_cells'][1:], ['100','120'])
        self.assertEqual(scoped['rows'][3]['display_cells'][1:], ['万元','万元'])
        slot=Slot('revenue',('营业收入',),'2024')
        self.assertTrue(table_binding_match(slot,scoped,'甲公司','甲公司','120','万元')['matched'])

    def test_company_row_year_table_without_metric_label_unchanged(self):
        original = table([['公司','2023','2024'],['甲公司','100','120'],['乙公司','70','80']],caption='营业收入')
        self.assertEqual(metric_scoped_table(original,'营业收入','甲公司'), original)

    def test_exact_out_of_vocabulary_metric_row_retained_without_borrowing(self):
        original = table([['项目','2023年','2022年'],['商誉','0.00','2.00'],
                          ['资产总计','100','120']], caption='甲公司；单位：元')
        scoped = metric_scoped_table(original,'商誉','甲公司')
        self.assertEqual([r['row'] for r in scoped['rows']], [1,2])
        self.assertEqual(scoped['content_text'], original['content_text'])
        slot = Slot('goodwill', ('商誉',), '2023')
        self.assertTrue(table_binding_match(slot,scoped,'甲公司','甲公司','0','元')['matched'])
        self.assertFalse(table_binding_match(slot,scoped,'甲公司','甲公司','100','元')['matched'])
        self.assertFalse(table_binding_match(slot,scoped,'甲公司','甲公司','2','元')['matched'])

    def test_exact_unknown_label_does_not_match_longer_financial_label(self):
        original = table([['项目','2023年','2022年'],['销售费用','5','6'],
                          ['其他销售费用','10','11'],['资产总计','100','120']])
        scoped = metric_scoped_table(original,'销售费用','甲公司')
        self.assertEqual([r['row'] for r in scoped['rows']], [1,2])

    def test_explicit_horizontal_groups_keep_metric_year_and_source_bytes(self):
        original=table([['公司','流动资产（三列）','非流动资产（三列）'],
            ['公司','2017年末','2018年末','2019年末','2017年末','2018年末','2019年末'],
            ['甲公司','91','95','100','201','210','220'],
            ['乙公司','12','15','18','25','28','31']],header=(1,2),data=[3,4],caption='单位：元')
        for metric,value,wrong,column in [('流动资产','100','220',3),('非流动资产','220','100',6)]:
            scoped=metric_scoped_table(original,metric,'甲公司')
            slot=Slot('amount',(metric,),'2019')
            self.assertTrue(table_binding_match(slot,scoped,'甲公司','甲公司',value,'元')['matched'])
            self.assertFalse(table_binding_match(slot,scoped,'甲公司','甲公司',wrong,'元')['matched'])
            self.assertEqual(scoped['parser_view_source_columns_v4'][-1],column)
            self.assertEqual(scoped['content_text'],original['content_text'])
        self.assertEqual(original['rows'][2]['display_cells'],['甲公司','91','95','100','201','210','220'])

    def test_horizontal_groups_with_inconsistent_periods_are_not_guessed(self):
        original=table([['公司','流动资产（三列）','非流动资产（三列）'],
            ['公司','2017年末','2018年末','2019年末','2016年末','2018年末','2019年末'],
            ['甲公司','91','95','100','201','210','220']],header=(1,2),data=[3])
        self.assertNotIn('explicit_horizontal_group_v4',metric_scoped_table(original,'流动资产','甲公司'))


if __name__=='__main__': unittest.main()

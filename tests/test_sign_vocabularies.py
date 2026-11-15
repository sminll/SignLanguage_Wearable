"""Synthetic mechanics only, excluded from experiment outputs."""
import unittest
from unittest.mock import patch
import pandas as pd
from tools import compare_sign_vocabularies as vocab
from tools import evaluate_three_sign_candidates as tri
from tools import evaluate_six_sign_knn as base


def expanded(count):
    from test_three_sign_candidates import fixture
    data=fixture()
    labels=tri.LABELS+['교통사고','마시다'][:count-3]
    extra=[]
    for who in ['subject1','subject2','subject3','subject4']:
        prototype=data[data.person.eq(who)].iloc[0].to_dict()
        for number,label in enumerate(labels[3:]):
            row=dict(prototype,label=label,trial_id=f'{who}-extra{number}',file=f'{who}-extra{number}.csv')
            for c in base.ALL_COLS:
                row[c]=float(500+number*100)
            extra.append(row)
    return pd.concat([data,pd.DataFrame(extra)],ignore_index=True),labels


class VocabularyTests(unittest.TestCase):
    def test_all_five_sets_keep_four_core_signs(self):
        specs=vocab.vocabularies()
        self.assertEqual(len(specs),7)
        self.assertEqual([len(r['labels']) for r in specs],[3,4,5,5,5,5,5])
        self.assertEqual({r['extra'] for r in specs if 'extra' in r},set(vocab.EXTRAS))
        for row in specs[2:]:
            self.assertEqual(row['labels'][:4],vocab.FOUR)

    def test_explicit_four_five_labels_keep_partial_people_without_global_mutation(self):
        defaults=list(tri.LABELS)
        configs=[dict(config_id=i,sensors=s,k=3,p=1,weights='uniform') for i,s in enumerate(base.FEATURES)]
        for count in [4,5]:
            data,labels=expanded(count)
            seen=[]

            def fake_inner(train,grid,labels=None):
                self.assertEqual(set(train.label),set(labels))
                seen.append(set(train.person))
                choices={c['sensors']:dict(**c,selection_score=0) for c in configs}
                choices['nested_selection']=choices['imu_only']
                return choices,[]

            with patch.object(tri,'select_inner',fake_inner):
                rows,_,_=tri.evaluate(data,configs,labels=labels)
            for who,fit_people in zip(base.people_order(data.person),seen):
                self.assertNotIn(who,fit_people)
            self.assertFalse(rows.duplicated(['method','file']).any())
            self.assertEqual(len(rows),len(data)*4)
            report=tri.summary(rows[rows.method.eq('nested_selection')],labels)
            self.assertEqual(report['labels'],labels)
            self.assertEqual(len(report['confusion_matrix']),count)
            self.assertEqual(sum(map(sum,report['confusion_matrix'])),len(data))
            self.assertEqual(tri.LABELS,defaults)

    def test_inner_scoring_uses_requested_five_labels(self):
        data,labels=expanded(5)
        configs=[dict(config_id=i,sensors=s,k=3,p=1,weights='uniform') for i,s in enumerate(base.FEATURES)]
        real=base.make_model
        checks=[]

        class Tracked:
            def __init__(self,c):
                self.model=real(c)
            def fit(self,x,y):
                self.people=set(data.loc[x.index,'person'])
                assert set(y)==set(labels)
                self.model.fit(x,y)
                return self
            def predict(self,x):
                assert self.people.isdisjoint(data.loc[x.index,'person'])
                checks.append(True)
                return self.model.predict(x)

        with patch.object(base,'make_model',Tracked):
            _,board=tri.select_inner(data,configs,labels)
        self.assertEqual(len(checks),12)
        self.assertTrue(all(len(c['inner_folds'])==4 for c in board))

    def test_selection_does_not_hide_weak_fifth_with_high_average(self):
        weak=dict(name='weak',minimum_sign_recall=0.0,macro_recall=.8,accuracy=.95)
        balanced=dict(name='balanced',minimum_sign_recall=.5,macro_recall=.65,accuracy=.65)
        self.assertEqual(vocab.choose_five([weak,balanced])['name'],'balanced')


if __name__=='__main__':
    unittest.main()

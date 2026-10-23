"""Fixed positive behavior weights; scientific failures reduce reward monotonically."""
from collections import Counter
from fractions import Fraction


def aggregate(nodeids,reports,profile):
    functions=[node.split('::')[-1].split('[')[0] for node in nodeids]
    counts=Counter(functions)
    weights=profile['weights']
    if set(counts)!=set(weights) or sum(weights.values())!=profile['total_weight']:
        raise ValueError('Scoring profile does not match the complete suite')
    earned=Fraction(0);breakdown=[]
    for node,function in zip(nodeids,functions):
        weight=Fraction(weights[function],counts[function])
        passed=reports[node]['call']['outcome']=='passed'
        contribution=weight if passed else Fraction(0)
        earned+=contribution
        breakdown.append({'nodeid':node,'weight':float(weight),'passed':passed,'earned':float(contribution)})
    return float(earned/profile['total_weight']),breakdown

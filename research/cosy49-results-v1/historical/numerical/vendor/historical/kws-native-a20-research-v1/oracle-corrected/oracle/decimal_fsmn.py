"""Independent Decimal80 post-CMVN ideal FSMN and conservative interval gate.

Inert module: no filesystem, Torch, native library, checkpoint or audio access.
Inputs/weights must be previously identity-verified NumPy float32 arrays.
Construction prepares constants only; evaluate() executes one mathematical row
per supplied CMVN row, with fresh cache. Nothing is rounded back to FP32.
"""
import decimal
import math

D = decimal.Decimal
ZERO, ONE = D(0), D(1)
PRECISION = 80
UNIT_ROUNDOFF = D('5e-80')
UNCERTAINTY_CAP = D('1e-20')
ATOL, RTOL = D('1e-4'), D('1e-5')
DIMS = [140,250,250] + [128,128,250,250]*4 + [140,6]


def _contexts():
    contexts = [decimal.Context(prec=PRECISION, rounding=mode, Emin=-999999, Emax=999999)
                for mode in [decimal.ROUND_HALF_EVEN, decimal.ROUND_CEILING, decimal.ROUND_FLOOR]]
    for context in contexts:
        for trap in [decimal.InvalidOperation, decimal.Overflow, decimal.Underflow,
                     decimal.Subnormal, decimal.DivisionByZero]:
            context.traps[trap] = True
    return contexts


def _float32(array, shape):
    assert getattr(array.dtype, 'name', None) == 'float32' and tuple(array.shape) == tuple(shape)
    assert all(math.isfinite(float(x)) for x in array.reshape(-1))


def _maximum_absolute(values):
    return max((x.copy_abs() for x in values), default=ZERO)


class Decimal80FSMN:
    """Exact FP32 constants, Decimal80 internal state, scalar rigorous enclosures."""
    def __init__(self, state):
        self.nearest, self.up, self.down = _contexts()
        self.gamma_cache, self.dense, self.mem = {}, {}, []
        definitions = [('backbone.in_linear1',400,140), ('backbone.in_linear2',140,250)]
        for layer in range(4):
            definitions += [(f'backbone.fsmn.{layer}.0',250,128),
                            (f'backbone.fsmn.{layer}.2',128,250)]
        definitions += [('backbone.out_linear1',250,140), ('backbone.out_linear2',140,6)]
        for stem, ni, no in definitions:
            a = state[stem+'.linear.weight']; _float32(a, (no,ni))
            flat = [D.from_float(float(x)) for x in a.reshape(-1)]
            matrix = [flat[o*ni:(o+1)*ni] for o in range(no)]
            if stem+'.linear.bias' in state:
                b = state[stem+'.linear.bias']; _float32(b, (no,))
                bias = [D.from_float(float(x)) for x in b]
            else:
                assert stem.endswith('.0'), 'Only FSMN projections omit bias'
                bias = [ZERO]*no
            norm = max(self._sum_up(x.copy_abs() for x in row) for row in matrix)
            self.dense[stem] = (matrix,bias,ni,no,norm,_maximum_absolute(bias))
        for layer in range(4):
            a=state[f'backbone.fsmn.{layer}.1.conv_left.weight']; _float32(a,(128,1,10,1))
            b=state[f'backbone.fsmn.{layer}.1.conv_right.weight']; _float32(b,(128,1,2,1))
            flat_a=[D.from_float(float(x)) for x in a.reshape(-1)]
            flat_b=[D.from_float(float(x)) for x in b.reshape(-1)]
            left=[flat_a[c*10:(c+1)*10] for c in range(128)]
            right=[flat_b[c*2:(c+1)*2] for c in range(128)]
            gain=max(self.up.add(ONE,self._sum_up(x.copy_abs() for x in left[c]+right[c]))
                     for c in range(128))
            self.mem.append((left,right,gain))

    def _sum_up(self, values):
        result=ZERO
        for value in values: result=self.up.add(result,value)
        return result

    def _gamma(self,k):
        if k not in self.gamma_cache:
            ku=self.up.multiply(D(k),UNIT_ROUNDOFF)
            denominator=self.down.subtract(ONE,ku)
            assert denominator>0
            self.gamma_cache[k]=self.up.divide(ku,denominator)
        return self.gamma_cache[k]

    def _affine(self,stem,x,error,counters):
        matrix,bias,ni,no,norm,bmax=self.dense[stem]
        assert len(x)==ni
        output=[]
        for row,b in zip(matrix,bias):
            total=ZERO
            for value,weight in zip(x,row): total=total+value*weight
            output.append(total+b)
        l1=self.up.add(self.up.multiply(_maximum_absolute(x),norm),bmax)
        e=self.up.add(self.up.multiply(error,norm),self.up.multiply(self._gamma(ni+1),l1))
        assert all(x.is_finite() for x in output) and e.is_finite()
        counters['affine_calls']+=1; counters['weighted_products']+=ni*no
        return output,e

    def evaluate(self,cmvn):
        """Return JSON-safe strings: logits/lowers/uppers [T][6], radii [T].

        Also returns stages [stage][T][width], per-stage scalar error bounds,
        final cache [layer][channel][11], and exact mathematical-work counts.
        Each call resets cache; dtype is decimal strings, not float64/object NPZ.
        """
        assert getattr(cmvn,'ndim',None)==2 and cmvn.shape[1]==400
        count=int(cmvn.shape[0]); assert 1<=count<=64
        _float32(cmvn,(count,400))
        with decimal.localcontext(self.nearest):
            caches=[[[ZERO]*11 for _ in range(128)] for _ in range(4)]
            cache_error=[[ZERO]*11 for _ in range(4)]
            cache_max=[[ZERO]*11 for _ in range(4)]
            stages={f'stage{i}':[] for i in range(21)}
            errors={f'stage{i}':[] for i in range(21)}
            counters={'reference_rows':0,'affine_calls':0,'memory_calls':0,'weighted_products':0}
            def keep(i,values,e):
                assert len(values)==DIMS[i]
                stages[f'stage{i}'].append([str(x) for x in values])
                errors[f'stage{i}'].append(str(e))
            def relu(x):return [value if value>=0 else ZERO for value in x]
            for row in cmvn:
                counters['reference_rows']+=1
                x=[D.from_float(float(v)) for v in row];e=ZERO
                x,e=self._affine('backbone.in_linear1',x,e,counters);keep(0,x,e)
                x,e=self._affine('backbone.in_linear2',x,e,counters);keep(1,x,e)
                x=relu(x);keep(2,x,e)
                for layer in range(4):
                    x,e=self._affine(f'backbone.fsmn.{layer}.0',x,e,counters);keep(3+4*layer,x,e)
                    projection,projection_error=x,e
                    projection_max=_maximum_absolute(projection)
                    left,right,gain=self.mem[layer]
                    input_error=max([projection_error,*cache_error[layer]])
                    input_max=max([projection_max,*cache_max[layer]])
                    output=[]
                    for c in range(128):
                        h=caches[layer][c];total=ZERO
                        for k in range(10):total=total+left[c][k]*h[k]
                        right_sum=right[c][0]*h[10]+right[c][1]*projection[c]
                        output.append((h[9]+total)+right_sum)
                        caches[layer][c]=h[1:]+[projection[c]]
                    cache_error[layer]=cache_error[layer][1:]+[projection_error]
                    cache_max[layer]=cache_max[layer][1:]+[projection_max]
                    e=self.up.add(self.up.multiply(input_error,gain),
                                  self.up.multiply(self._gamma(14),self.up.multiply(input_max,gain)))
                    x=output;assert all(v.is_finite() for v in x) and e.is_finite()
                    counters['memory_calls']+=1;counters['weighted_products']+=128*12
                    keep(4+4*layer,x,e)
                    x,e=self._affine(f'backbone.fsmn.{layer}.2',x,e,counters);keep(5+4*layer,x,e)
                    x=relu(x);keep(6+4*layer,x,e)
                x,e=self._affine('backbone.out_linear1',x,e,counters);keep(19,x,e)
                x,e=self._affine('backbone.out_linear2',x,e,counters);keep(20,x,e)
            assert counters=={'reference_rows':count,'affine_calls':12*count,
                              'memory_calls':4*count,'weighted_products':388984*count}
            lower,upper,radii=[],[],[]
            for values,e_string in zip(stages['stage20'],errors['stage20']):
                centers=[D(s) for s in values];e=D(e_string)
                low=[self.down.subtract(h,e) for h in centers]
                high=[self.up.add(h,e) for h in centers]
                radius=max(max(self.up.subtract(h,l),self.up.subtract(u,h))
                           for h,l,u in zip(centers,low,high))
                assert radius<=UNCERTAINTY_CAP, 'Oracle uncertainty exceeds predeclared cap'
                lower.append([str(v) for v in low]);upper.append([str(v) for v in high]);radii.append(str(radius))
            return {'schema':'a20-decimal80-ideal-result.v1','precision_digits':80,
                    'input_rows':count,'logits':stages['stage20'],'lower':lower,'upper':upper,
                    'error_bounds':errors['stage20'],'interval_radii':radii,
                    'uncertainty_cap':str(UNCERTAINTY_CAP),'stages':stages,'stage_error_bounds':errors,
                    'final_cache':[[[str(x) for x in h] for h in layer] for layer in caches],
                    'counters':counters,'oracle_definition':'Exact FP32 weights and fixed FP32 CMVN inputs; Decimal80 intermediates and cache without FP32 boundary casts',
                    'rounding_error_model':'Nearest Decimal80 unit roundoff5e-80, upward local and propagated row-L1/cache error bounds; no overflow/subnormal/underflow permitted'}


def compare_float32_logits(actual,ideal):
    """Certify unchanged raw-logit tolerance for every true value in interval.

    Worst endpoint distance is rounded up. Relative-term minimum magnitude and
    the tolerance are rounded down. No uncertainty is added to the tolerance.
    A failed certification fails the hard numerical gate; no auto rerun occurs.
    """
    rows=int(ideal['input_rows']);assert 1<=rows<=64;_float32(actual,(rows,6))
    assert ideal['precision_digits']==80 and D(ideal['uncertainty_cap'])==UNCERTAINTY_CAP
    _,up,down=_contexts();failed=[];max_distance=ZERO;max_ratio=ZERO;min_margin=None
    for i in range(rows):
        assert len(ideal['logits'][i])==len(ideal['lower'][i])==len(ideal['upper'][i])==6
        assert ZERO<=D(ideal['interval_radii'][i])<=UNCERTAINTY_CAP
        for j in range(6):
            center=D(ideal['logits'][i][j]);lo=D(ideal['lower'][i][j]);hi=D(ideal['upper'][i][j])
            assert all(v.is_finite() for v in [center,lo,hi]) and lo<=center<=hi
            assert max(up.subtract(center,lo),up.subtract(hi,center))<=UNCERTAINTY_CAP
            value=D.from_float(float(actual[i,j]))
            distance=max(up.subtract(value,lo),up.subtract(hi,value))
            minimum_magnitude=ZERO if lo<=0<=hi else min(lo.copy_abs(),hi.copy_abs())
            tolerance=down.add(ATOL,down.multiply(RTOL,minimum_magnitude))
            ratio=up.divide(distance,tolerance);margin=down.subtract(tolerance,distance)
            max_distance=max(max_distance,distance);max_ratio=max(max_ratio,ratio)
            min_margin=margin if min_margin is None else min(min_margin,margin)
            if distance>tolerance:
                failed.append({'index':[i,j],'actual':str(value),'oracle_center':str(center),
                               'lower':str(lo),'upper':str(hi),'worst_absolute_error_upper':str(distance),
                               'tolerance_lower':str(tolerance),'margin_lower':str(margin)})
    return {'passed':not failed,'elements':rows*6,'failed_elements':len(failed),
            'maximum_absolute_error_upper':str(max_distance),'maximum_tolerance_ratio_upper':str(max_ratio),
            'minimum_margin_lower':str(min_margin),'failures':failed,
            'atol':'1e-4','rtol':'1e-5','oracle_uncertainty_cap':'1e-20',
            'criterion':'up(max(actual-lower,upper-actual)) <= down(1e-4+1e-5*min_abs_interval)'}


def slice_interval_result(ideal,offset,count):
    """Small JSON-safe view for a route chunk; no new mathematical evaluation."""
    assert isinstance(offset,int) and isinstance(count,int) and count>0
    assert 0<=offset<ideal['input_rows'] and offset+count<=ideal['input_rows']
    result={key:ideal[key] for key in ['schema','precision_digits','uncertainty_cap']}
    result['input_rows']=count
    for key in ['logits','lower','upper','error_bounds','interval_radii']:
        assert len(ideal[key])==ideal['input_rows']
        result[key]=ideal[key][offset:offset+count]
    return result

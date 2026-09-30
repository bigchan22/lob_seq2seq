def round_robin_shards(assets):
 assets=sorted(assets);return {'A':assets[::2],'B':assets[1::2]}
def validate_shards(shards,expected=27):
 flat=sum(shards.values(),[])
 if len(flat)!=expected or len(set(flat))!=expected:raise ValueError('S shards must contain exactly 27 unique assets')
 return True

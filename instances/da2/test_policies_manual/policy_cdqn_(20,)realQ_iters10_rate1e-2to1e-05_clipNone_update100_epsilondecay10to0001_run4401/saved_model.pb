цВ
н¬
Ь
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
≥
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8зн
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
д
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*U
shared_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
Ё
XCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:
*
dtype0
№
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
’
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOpBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias*
_output_shapes
:*
dtype0
…
6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ы0*G
shared_name86CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel
¬
JCategoricalQNetwork/CategoricalQNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel*
_output_shapes
:	Ы0*
dtype0
Ѕ
4CategoricalQNetwork/CategoricalQNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ы0*E
shared_name64CategoricalQNetwork/CategoricalQNetwork/dense_1/bias
Ї
HCategoricalQNetwork/CategoricalQNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp4CategoricalQNetwork/CategoricalQNetwork/dense_1/bias*
_output_shapes	
:Ы0*
dtype0
Ю
ConstConst*
_output_shapes
:3*
dtype0*д
valueЏB„3"ћ   ЅЪЩЅ33ЅЌћЅffЅ   Ѕ33ујffжјЪЩўјЌћћј  јј33≥јff¶јЪЩЩјЌћМј  АјfffјЌћLј333јЪЩј   јЌћћњЪЩЩњЌћLњЌћћЊ    Ќћћ>ЌћL?ЪЩЩ?Ќћћ?   @ЪЩ@333@ЌћL@fff@  А@ЌћМ@ЪЩЩ@ff¶@33≥@  ј@Ќћћ@ЪЩў@ffж@33у@   AffAЌћA33AЪЩA   A

NoOpNoOp
≤
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*л
valueбBё B„
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
	3


0
 
ЗД
VARIABLE_VALUEDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE4CategoricalQNetwork/CategoricalQNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE

ref
1


_q_network
b

_q_network
regularization_losses
trainable_variables
	variables
	keras_api
t
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
 

0
1
2
	3

0
1
2
	3
≠
regularization_losses
layer_metrics
trainable_variables
layer_regularization_losses
non_trainable_variables
	variables
metrics

layers
n
_postprocessing_layers
regularization_losses
trainable_variables
 	variables
!	keras_api
h

kernel
	bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
 

0
1
2
	3

0
1
2
	3
≠
regularization_losses
&layer_metrics
trainable_variables
'layer_regularization_losses
(non_trainable_variables
	variables
)metrics

*layers
 
 
 
 

0

+0
,1
 

0
1

0
1
≠
regularization_losses
-layer_metrics
trainable_variables
.layer_regularization_losses
/non_trainable_variables
 	variables
0metrics

1layers
 

0
	1

0
	1
≠
"regularization_losses
2layer_metrics
#trainable_variables
3layer_regularization_losses
4non_trainable_variables
$	variables
5metrics

6layers
 
 
 
 

0
1
R
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

kernel
bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
 
 
 
 

+0
,1
 
 
 
 
 
 
 
 
≠
7regularization_losses
?layer_metrics
8trainable_variables
@layer_regularization_losses
Anon_trainable_variables
9	variables
Bmetrics

Clayers
 

0
1

0
1
≠
;regularization_losses
Dlayer_metrics
<trainable_variables
Elayer_regularization_losses
Fnon_trainable_variables
=	variables
Gmetrics

Hlayers
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
w
action_0/observationPlaceholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape:€€€€€€€€€

j
action_0/rewardPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
m
action_0/step_typePlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
п
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_typeDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_1/biasConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1235720
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ъ
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1235732
џ
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1235754
Ц
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1235747
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpXCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpVCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpJCategoricalQNetwork/CategoricalQNetwork/dense_1/kernel/Read/ReadVariableOpHCategoricalQNetwork/CategoricalQNetwork/dense_1/bias/Read/ReadVariableOpConst_1*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_1235950
Ю
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_1235975Ю≠
Є

џ
+__inference_function_with_signature_1235700
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *2
f-R+
)__inference_polymorphic_action_fn_12356872
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
:::::322
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:VR
'
_output_shapes
:€€€€€€€€€

'
_user_specified_name0/observation: 

_output_shapes
:3
«
'
%__inference_signature_wrapper_1235754И
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *4
f/R-
+__inference_function_with_signature_12357502
PartitionedCall*
_input_shapes 
Г
_
%__inference_signature_wrapper_1235747
unknown
identity	ИҐStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *4
f/R-
+__inference_function_with_signature_12357392
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
ш
e
+__inference_function_with_signature_1235739
unknown
identity	ИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *#
fR
__inference_<lambda>_489792
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
Ф
7
%__inference_get_initial_state_1235726

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
СW
µ
)__inference_polymorphic_action_fn_1235687
	time_step
time_step_1
time_step_2
time_step_3`
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resourcea
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceR
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resourceS
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource	
mul_x
identityИҐTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpҐECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpя
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Constђ
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3NCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape«
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpч
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul∆
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddТ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhЮ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Ы0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpƒ
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЭ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ы0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp¬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЫ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€y   3   2#
!CategoricalQNetwork/Reshape/shapeй
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:€€€€€€€€€y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€y2
SumХ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimension™
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtolН
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeЗ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeХ
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1“
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/ConstЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis®
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatќ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1Ґ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceО
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisИ
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1–
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :x2
clip_by_value/Minimum/y≤
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yМ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_valueЯ
IdentityIdentityclip_by_value:z:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
:::::32ђ
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2™
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2О
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:NJ
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	time_step:RN
'
_output_shapes
:€€€€€€€€€

#
_user_specified_name	time_step: 

_output_shapes
:3
ё
£
#__inference__traced_restore_1235975
file_prefix
assignvariableop_variable[
Wassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernelY
Uassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasM
Iassignvariableop_3_categoricalqnetwork_categoricalqnetwork_dense_1_kernelK
Gassignvariableop_4_categoricalqnetwork_categoricalqnetwork_dense_1_bias

identity_6ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4Ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Т
valueИBЕB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЪ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices…
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

IdentityШ
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1№
AssignVariableOp_1AssignVariableOpWassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Џ
AssignVariableOp_2AssignVariableOpUassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ќ
AssignVariableOp_3AssignVariableOpIassignvariableop_3_categoricalqnetwork_categoricalqnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ћ
AssignVariableOp_4AssignVariableOpGassignvariableop_4_categoricalqnetwork_categoricalqnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpѕ

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5Ѕ

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
1

__inference_<lambda>_48982*
_input_shapes 
Ф
7
%__inference_get_initial_state_1235906

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
∆
М
 __inference__traced_save_1235950
file_prefix'
#savev2_variable_read_readvariableop	c
_savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel_read_readvariableopa
]savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias_read_readvariableopU
Qsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_kernel_read_readvariableopS
Osavev2_categoricalqnetwork_categoricalqnetwork_dense_1_bias_read_readvariableop
savev2_const_1

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameА
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Т
valueИBЕB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesФ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop_savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel_read_readvariableop]savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias_read_readvariableopQsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_kernel_read_readvariableopOsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_bias_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*;
_input_shapes*
(: : :
::	Ы0:Ы0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
::%!

_output_shapes
:	Ы0:!

_output_shapes	
:Ы0:

_output_shapes
: 
Љ
-
+__inference_function_with_signature_1235750ч
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *#
fR
__inference_<lambda>_489822
PartitionedCall*
_input_shapes 
Ѓ
7
%__inference_signature_wrapper_1235732

batch_sizeЧ
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *4
f/R-
+__inference_function_with_signature_12357272
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Ѓ
=
+__inference_function_with_signature_1235727

batch_sizeС
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_get_initial_state_12357262
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
бW
’
)__inference_polymorphic_action_fn_1235810
time_step_step_type
time_step_reward
time_step_discount
time_step_observation`
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resourcea
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceR
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resourceS
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource	
mul_x
identityИҐTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpҐECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpя
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Constґ
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape«
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpч
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul∆
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddТ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhЮ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Ы0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpƒ
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЭ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ы0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp¬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЫ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€y   3   2#
!CategoricalQNetwork/Reshape/shapeй
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:€€€€€€€€€y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€y2
SumХ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimension™
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtolН
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeЗ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeХ
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1“
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/ConstЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis®
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatќ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1Ґ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceО
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisИ
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1–
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :x2
clip_by_value/Minimum/y≤
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yМ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_valueЯ
IdentityIdentityclip_by_value:z:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
:::::32ђ
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2™
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2О
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:X T
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:€€€€€€€€€

/
_user_specified_nametime_step/observation: 

_output_shapes
:3
ЗW
≠
)__inference_polymorphic_action_fn_1235865
	step_type

reward
discount
observation`
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resourcea
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceR
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resourceS
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource	
mul_x
identityИҐTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpҐECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpя
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Constђ
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapeobservationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape«
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpч
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul∆
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddТ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhЮ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Ы0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpƒ
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЭ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ы0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp¬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЫ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€y   3   2#
!CategoricalQNetwork/Reshape/shapeй
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:€€€€€€€€€y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€y2
SumХ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimension™
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtolН
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeЗ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeХ
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1“
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/ConstЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis®
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatќ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1Ґ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceО
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisИ
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1–
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :x2
clip_by_value/Minimum/y≤
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yМ
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
clip_by_valueЯ
IdentityIdentityclip_by_value:z:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
:::::32ђ
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2™
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2О
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:TP
'
_output_shapes
:€€€€€€€€€

%
_user_specified_nameobservation: 

_output_shapes
:3
Ш=
≥
/__inference_polymorphic_distribution_fn_1235903
	step_type

reward
discount
observation`
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resourcea
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceR
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resourceS
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource	
mul_x
identityИҐTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpҐECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpя
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Constђ
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapeobservationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape«
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpч
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul∆
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddТ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhЮ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	Ы0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpƒ
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЭ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ы0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp¬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Ы029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЫ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€y   3   2#
!CategoricalQNetwork/Reshape/shapeй
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:€€€€€€€€€y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€y2
SumХ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2%
#Categorical_1/mode/ArgMax/dimension™
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/ArgMaxЫ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:€€€€€€€€€2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtol©
IdentityIdentityCategorical_1/mode/Cast:y:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:€€€€€€€€€2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0*i
_input_shapesX
V:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
:::::32ђ
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2™
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2Р
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2О
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	step_type:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_namereward:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
discount:TP
'
_output_shapes
:€€€€€€€€€

%
_user_specified_nameobservation: 

_output_shapes
:3
Х
[
__inference_<lambda>_48979
readvariableop_resource
identity	ИҐReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2 
ReadVariableOpReadVariableOp
і

’
%__inference_signature_wrapper_1235720
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *4
f/R-
+__inference_function_with_signature_12357002
StatefulPartitionedCallК
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€:::::322
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
0/discount:VR
'
_output_shapes
:€€€€€€€€€

'
_user_specified_name0/observation:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
0/reward:PL
#
_output_shapes
:€€€€€€€€€
%
_user_specified_name0/step_type: 

_output_shapes
:3"±L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*њ
actionі
4

0/discount&
action_0/discount:0€€€€€€€€€
>
0/observation-
action_0/observation:0€€€€€€€€€

0
0/reward$
action_0/reward:0€€€€€€€€€
6
0/step_type'
action_0/step_type:0€€€€€€€€€6
action,
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:ґv
Ќ

train_step
metadata
model_variables
_all_assets

signatures

Iaction
Jdistribution
Kget_initial_state
Lget_metadata
Mget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
=
0
1
2
	3"
trackable_tuple_wrapper
'

0"
trackable_list_wrapper
`

Naction
Oget_initial_state
Pget_train_step
Qget_metadata"
signature_map
V:T
2DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
P:N2BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
I:G	Ы026CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel
C:AЫ024CategoricalQNetwork/CategoricalQNetwork/dense_1/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
«

_q_network
regularization_losses
trainable_variables
	variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"®
_tf_keras_layerО{"class_name": "CategoricalQNetwork", "name": "CategoricalQNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ќ
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "QNetwork", "name": "CategoricalQNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
≠
regularization_losses
layer_metrics
trainable_variables
layer_regularization_losses
non_trainable_variables
	variables
metrics

layers
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
Ћ
_postprocessing_layers
regularization_losses
trainable_variables
 	variables
!	keras_api
V__call__
*W&call_and_return_all_conditional_losses"†
_tf_keras_layerЖ{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ґ

kernel
	bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"э
_tf_keras_layerг{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 6171, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20]}}
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
≠
regularization_losses
&layer_metrics
trainable_variables
'layer_regularization_losses
(non_trainable_variables
	variables
)metrics

*layers
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
regularization_losses
-layer_metrics
trainable_variables
.layer_regularization_losses
/non_trainable_variables
 	variables
0metrics

1layers
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
≠
"regularization_losses
2layer_metrics
#trainable_variables
3layer_regularization_losses
4non_trainable_variables
$	variables
5metrics

6layers
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
в
7regularization_losses
8trainable_variables
9	variables
:	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"”
_tf_keras_layerє{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
√

kernel
bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
\__call__
*]&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 10]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
7regularization_losses
?layer_metrics
8trainable_variables
@layer_regularization_losses
Anon_trainable_variables
9	variables
Bmetrics

Clayers
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
;regularization_losses
Dlayer_metrics
<trainable_variables
Elayer_regularization_losses
Fnon_trainable_variables
=	variables
Gmetrics

Hlayers
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Н2К
)__inference_polymorphic_action_fn_1235810
)__inference_polymorphic_action_fn_1235865±
™≤¶
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsҐ
Ґ 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
и2е
/__inference_polymorphic_distribution_fn_1235903±
™≤¶
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsҐ
Ґ 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
%__inference_get_initial_state_1235906¶
Э≤Щ
FullArgSpec!
argsЪ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
∞B≠
__inference_<lambda>_48982"О
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
∞B≠
__inference_<lambda>_48979"О
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
%__inference_signature_wrapper_1235720
0/discount0/observation0/reward0/step_type"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕBћ
%__inference_signature_wrapper_1235732
batch_size"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЅBЊ
%__inference_signature_wrapper_1235747"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЅBЊ
%__inference_signature_wrapper_1235754"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
	J
Const9
__inference_<lambda>_48979Ґ

Ґ 
™ "К 	2
__inference_<lambda>_48982Ґ

Ґ 
™ "™ R
%__inference_get_initial_state_1235906)"Ґ
Ґ
К

batch_size 
™ "Ґ Т
)__inference_polymorphic_action_fn_1235810д	^ЖҐВ
ъҐц
о≤к
TimeStep6
	step_type)К&
time_step/step_type€€€€€€€€€0
reward&К#
time_step/reward€€€€€€€€€4
discount(К%
time_step/discount€€€€€€€€€>
observation/К,
time_step/observation€€€€€€€€€

Ґ 
™ "R≤O

PolicyStep&
actionК
action€€€€€€€€€
stateҐ 
infoҐ к
)__inference_polymorphic_action_fn_1235865Љ	^ёҐЏ
“Ґќ
∆≤¬
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€4
observation%К"
observation€€€€€€€€€

Ґ 
™ "R≤O

PolicyStep&
actionК
action€€€€€€€€€
stateҐ 
infoҐ п
/__inference_polymorphic_distribution_fn_1235903ї	^ёҐЏ
“Ґќ
∆≤¬
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€4
observation%К"
observation€€€€€€€€€

Ґ 
™ "–≤ћ

PolicyStepҐ
actionЧТУрб√Г}Ґz
`
CҐ@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*™'
%
locК
Identity€€€€€€€€€
™ _TFPTypeSpec
stateҐ 
infoҐ є
%__inference_signature_wrapper_1235720П	^ЎҐ‘
Ґ 
ћ™»
.

0/discount К

0/discount€€€€€€€€€
8
0/observation'К$
0/observation€€€€€€€€€

*
0/rewardК
0/reward€€€€€€€€€
0
0/step_type!К
0/step_type€€€€€€€€€"+™(
&
actionК
action€€€€€€€€€`
%__inference_signature_wrapper_123573270Ґ-
Ґ 
&™#
!

batch_sizeК

batch_size "™ Y
%__inference_signature_wrapper_12357470Ґ

Ґ 
™ "™

int64К
int64 	=
%__inference_signature_wrapper_1235754Ґ

Ґ 
™ "™ 
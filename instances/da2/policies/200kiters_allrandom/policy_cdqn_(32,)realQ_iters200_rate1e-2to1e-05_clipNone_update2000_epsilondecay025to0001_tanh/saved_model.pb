ЃІ
Э╚
ю
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
dtypetypeѕ
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
│
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
ї
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718┘ш
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
С
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 *U
shared_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
П
XCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:
 *
dtype0
▄
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
Н
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOpBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias*
_output_shapes
: *
dtype0
╔
6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 Џ0*G
shared_name86CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel
┬
JCategoricalQNetwork/CategoricalQNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel*
_output_shapes
:	 Џ0*
dtype0
┴
4CategoricalQNetwork/CategoricalQNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Џ0*E
shared_name64CategoricalQNetwork/CategoricalQNetwork/dense_1/bias
║
HCategoricalQNetwork/CategoricalQNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp4CategoricalQNetwork/CategoricalQNetwork/dense_1/bias*
_output_shapes	
:Џ0*
dtype0
ъ
ConstConst*
_output_shapes
:3*
dtype0*С
value┌BО3"╠  └└ВQИ└ОБ░└├ше└«GА└џЎЎ└ЁвЉ└q=і└\Јѓ└Ј┬u└fff└=
W└«G└ВQ8└├ш(└џЎ└q=
└Ј┬ш┐=
О┐ВQИ┐џЎЎ┐Ј┬u┐ВQ8┐Ј┬шЙЈ┬uЙ    Ј┬u>Ј┬ш>ВQ8?Ј┬u?џЎЎ?ВQИ?=
О?Ј┬ш?q=
@џЎ@├ш(@ВQ8@«G@=
W@fff@Ј┬u@\Јѓ@q=і@ЁвЉ@џЎЎ@«GА@├ше@ОБ░@ВQИ@  └@

NoOpNoOp
▓
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*в
valueрBя BО
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
Єё
VARIABLE_VALUEDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
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
	variables
trainable_variables
regularization_losses
	keras_api
t
_encoder
_q_value_layer
	variables
trainable_variables
regularization_losses
	keras_api

0
1
2
	3

0
1
2
	3
 
Г
non_trainable_variables

layers
metrics
	variables
layer_regularization_losses
trainable_variables
regularization_losses
layer_metrics
n
_postprocessing_layers
	variables
trainable_variables
 regularization_losses
!	keras_api
h

kernel
	bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api

0
1
2
	3

0
1
2
	3
 
Г
&non_trainable_variables

'layers
(metrics
	variables
)layer_regularization_losses
trainable_variables
regularization_losses
*layer_metrics
 

0
 
 
 

+0
,1

0
1

0
1
 
Г
-non_trainable_variables

.layers
/metrics
	variables
0layer_regularization_losses
trainable_variables
 regularization_losses
1layer_metrics

0
	1

0
	1
 
Г
2non_trainable_variables

3layers
4metrics
"	variables
5layer_regularization_losses
#trainable_variables
$regularization_losses
6layer_metrics
 

0
1
 
 
 
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
h

kernel
bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
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
 
 
 
Г
?non_trainable_variables

@layers
Ametrics
7	variables
Blayer_regularization_losses
8trainable_variables
9regularization_losses
Clayer_metrics

0
1

0
1
 
Г
Dnon_trainable_variables

Elayers
Fmetrics
;	variables
Glayer_regularization_losses
<trainable_variables
=regularization_losses
Hlayer_metrics
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
:         *
dtype0*
shape:         
w
action_0/observationPlaceholder*'
_output_shapes
:         
*
dtype0*
shape:         

j
action_0/rewardPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
m
action_0/step_typePlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
­
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_typeDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_1/biasConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ */
f*R(
&__inference_signature_wrapper_23400598
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ч
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
GPU 2J 8ѓ */
f*R(
&__inference_signature_wrapper_23400610
▄
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
GPU 2J 8ѓ */
f*R(
&__inference_signature_wrapper_23400632
Ќ
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
GPU 2J 8ѓ */
f*R(
&__inference_signature_wrapper_23400625
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
і
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_save_23400828
Ъ
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
GPU 2J 8ѓ *-
f(R&
$__inference__traced_restore_23400853іх
з
(
&__inference_signature_wrapper_23400632Ѕ
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
GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_234006282
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
┐
8
&__inference_get_initial_state_23400784

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ТW
У
*__inference_polymorphic_action_fn_23400565
	time_step
time_step_1
time_step_2
time_step_3n
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:
 k
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource: a
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource:	 Џ0^
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource:	Џ0	
mul_x
identityѕбTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpбSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpбFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpбECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp▀
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Constг
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3NCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeК
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpэ
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulк
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddњ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanhъ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 Џ0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp─
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЮ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Џ0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp┬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЏ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    y   3   2#
!CategoricalQNetwork/Reshape/shapeж
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:         y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:         y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:         y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         y2
SumЋ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#Categorical_1/mode/ArgMax/dimensionф
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         2
Categorical_1/mode/ArgMaxЏ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
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
Deterministic/rtolЇ
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeЄ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeЋ
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1м
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Constџ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0і
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisе
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat╬
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:         2$
"Deterministic_1/sample/BroadcastToЏ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1б
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackд
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1д
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2Ж
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceј
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisѕ
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1л
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:         2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :x2
clip_by_value/Minimum/y▓
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yї
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         2
clip_by_valueЪ
IdentityIdentityclip_by_value:z:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         :         :         :         
: : : : :32г
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2ф
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2љ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2ј
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:RN
'
_output_shapes
:         

#
_user_specified_name	time_step: 

_output_shapes
:3
▄W
Я
*__inference_polymorphic_action_fn_23400688
	step_type

reward
discount
observationn
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:
 k
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource: a
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource:	 Џ0^
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource:	Џ0	
mul_x
identityѕбTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpбSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpбFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpбECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp▀
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Constг
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapeobservationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeК
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpэ
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulк
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddњ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanhъ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 Џ0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp─
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЮ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Џ0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp┬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЏ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    y   3   2#
!CategoricalQNetwork/Reshape/shapeж
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:         y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:         y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:         y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         y2
SumЋ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#Categorical_1/mode/ArgMax/dimensionф
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         2
Categorical_1/mode/ArgMaxЏ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
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
Deterministic/rtolЇ
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeЄ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeЋ
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1м
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Constџ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0і
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisе
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat╬
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:         2$
"Deterministic_1/sample/BroadcastToЏ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1б
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackд
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1д
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2Ж
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceј
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisѕ
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1л
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:         2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :x2
clip_by_value/Minimum/y▓
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yї
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         2
clip_by_valueЪ
IdentityIdentityclip_by_value:z:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         :         :         :         
: : : : :32г
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2ф
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2љ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2ј
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	step_type:KG
#
_output_shapes
:         
 
_user_specified_namereward:MI
#
_output_shapes
:         
"
_user_specified_name
discount:TP
'
_output_shapes
:         

%
_user_specified_nameobservation: 

_output_shapes
:3
ј
ј
,__inference_function_with_signature_23400578
	step_type

reward
discount
observation
unknown:
 
	unknown_0: 
	unknown_1:	 Џ0
	unknown_2:	Џ0
	unknown_3
identityѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *3
f.R,
*__inference_polymorphic_action_fn_234005652
StatefulPartitionedCallі
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         :         :         :         
: : : : :322
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:         
%
_user_specified_name0/step_type:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:OK
#
_output_shapes
:         
$
_user_specified_name
0/discount:VR
'
_output_shapes
:         

'
_user_specified_name0/observation: 

_output_shapes
:3
┐
8
&__inference_get_initial_state_23400604

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
К
Ї
!__inference__traced_save_23400828
file_prefix'
#savev2_variable_read_readvariableop	c
_savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel_read_readvariableopa
]savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias_read_readvariableopU
Qsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_kernel_read_readvariableopS
Osavev2_categoricalqnetwork_categoricalqnetwork_dense_1_bias_read_readvariableop
savev2_const_1

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameђ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*њ
valueѕBЁB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesћ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices╩
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop_savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel_read_readvariableop]savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias_read_readvariableopQsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_kernel_read_readvariableopOsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_bias_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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
 : :	 Џ0:Џ0: 2(
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
 : 

_output_shapes
: :%!

_output_shapes
:	 Џ0:!

_output_shapes	
:Џ0:

_output_shapes
: 
┌
8
&__inference_signature_wrapper_23400610

batch_sizeў
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
GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_234006052
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
у
.
,__inference_function_with_signature_23400628э
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
GPU 2J 8ѓ *#
fR
__inference_<lambda>_461022
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
│
f
&__inference_signature_wrapper_23400625
unknown:	 
identity	ѕбStatefulPartitionedCallх
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
GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_234006172
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
ХX
ѕ
*__inference_polymorphic_action_fn_23400743
time_step_step_type
time_step_reward
time_step_discount
time_step_observationn
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:
 k
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource: a
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource:	 Џ0^
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource:	Џ0	
mul_x
identityѕбTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpбSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpбFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpбECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp▀
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstХ
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeК
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpэ
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulк
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddњ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanhъ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 Џ0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp─
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЮ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Џ0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp┬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЏ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    y   3   2#
!CategoricalQNetwork/Reshape/shapeж
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:         y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:         y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:         y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         y2
SumЋ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#Categorical_1/mode/ArgMax/dimensionф
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         2
Categorical_1/mode/ArgMaxЏ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
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
Deterministic/rtolЇ
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shapeЄ
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeЋ
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1м
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Constџ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0і
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisе
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat╬
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:         2$
"Deterministic_1/sample/BroadcastToЏ
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1б
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackд
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1д
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2Ж
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceј
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisѕ
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1л
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:         2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :x2
clip_by_value/Minimum/y▓
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yї
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         2
clip_by_valueЪ
IdentityIdentityclip_by_value:z:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         :         :         :         
: : : : :32г
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2ф
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2љ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2ј
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:X T
#
_output_shapes
:         
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:         
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:         
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:         

/
_user_specified_nametime_step/observation: 

_output_shapes
:3
Ї
▄
$__inference__traced_restore_23400853
file_prefix#
assignvariableop_variable:	 i
Wassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel:
 c
Uassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias: \
Iassignvariableop_3_categoricalqnetwork_categoricalqnetwork_dense_1_kernel:	 Џ0V
Gassignvariableop_4_categoricalqnetwork_categoricalqnetwork_dense_1_bias:	Џ0

identity_6ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4є
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*њ
valueѕBЁB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesџ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices╔
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

Identityў
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1▄
AssignVariableOp_1AssignVariableOpWassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2┌
AssignVariableOp_2AssignVariableOpUassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╬
AssignVariableOp_3AssignVariableOpIassignvariableop_3_categoricalqnetwork_categoricalqnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╠
AssignVariableOp_4AssignVariableOpGassignvariableop_4_categoricalqnetwork_categoricalqnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¤

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5┴

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*
_input_shapes
: : : : : : 2$
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
├
a
__inference_<lambda>_46099!
readvariableop_resource:	 
identity	ѕбReadVariableOpp
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
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
і
ѕ
&__inference_signature_wrapper_23400598
discount
observation

reward
	step_type
unknown:
 
	unknown_0: 
	unknown_1:	 Џ0
	unknown_2:	Џ0
	unknown_3
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *5
f0R.
,__inference_function_with_signature_234005782
StatefulPartitionedCallі
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         :         
:         :         : : : : :322
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:         
$
_user_specified_name
0/discount:VR
'
_output_shapes
:         

'
_user_specified_name0/observation:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:PL
#
_output_shapes
:         
%
_user_specified_name0/step_type: 

_output_shapes
:3
Д
l
,__inference_function_with_signature_23400617
unknown:	 
identity	ѕбStatefulPartitionedCallБ
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
GPU 2J 8ѓ *#
fR
__inference_<lambda>_460992
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
┌
>
,__inference_function_with_signature_23400605

batch_sizeњ
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
GPU 2J 8ѓ */
f*R(
&__inference_get_initial_state_234006042
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
[

__inference_<lambda>_46102*(
_construction_contextkEagerRuntime*
_input_shapes 
ь=
Т
0__inference_polymorphic_distribution_fn_23400781
	step_type

reward
discount
observationn
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource:
 k
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource: a
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource:	 Џ0^
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource:	Џ0	
mul_x
identityѕбTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpбSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpбFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpбECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp▀
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Constг
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapeobservationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         
2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeК
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpэ
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulк
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpщ
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddњ
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/TanhTanhNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanhъ
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes
:	 Џ0*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp─
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulFCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Tanh:y:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ028
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulЮ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Џ0*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp┬
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Џ029
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddЏ
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"    y   3   2#
!CategoricalQNetwork/Reshape/shapeж
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:         y32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:         y32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:         y32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:         y2
SumЋ
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#Categorical_1/mode/ArgMax/dimensionф
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         2
Categorical_1/mode/ArgMaxЏ
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         2
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
Deterministic_1/rtolЕ
IdentityIdentityCategorical_1/mode/Cast:y:0U^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpT^CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpG^CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpF^CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp*
T0*#
_output_shapes
:         2

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
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         :         :         :         
: : : : :32г
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpTCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2ф
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpSCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2љ
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpFCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp2ј
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:N J
#
_output_shapes
:         
#
_user_specified_name	step_type:KG
#
_output_shapes
:         
 
_user_specified_namereward:MI
#
_output_shapes
:         
"
_user_specified_name
discount:TP
'
_output_shapes
:         

%
_user_specified_nameobservation: 

_output_shapes
:3"╠L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
action┤
4

0/discount&
action_0/discount:0         
>
0/observation-
action_0/observation:0         

0
0/reward$
action_0/reward:0         
6
0/step_type'
action_0/step_type:0         6
action,
StatefulPartitionedCall:0         tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:кv
═
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
 2DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
P:N 2BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
I:G	 Џ026CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel
C:AЏ024CategoricalQNetwork/CategoricalQNetwork/dense_1/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
К

_q_network
	variables
trainable_variables
regularization_losses
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"е
_tf_keras_layerј{"name": "CategoricalQNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "CategoricalQNetwork", "config": {"layer was saved without config": true}}
╬
_encoder
_q_value_layer
	variables
trainable_variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"Ю
_tf_keras_layerЃ{"name": "CategoricalQNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}}
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
 "
trackable_list_wrapper
Г
non_trainable_variables

layers
metrics
	variables
layer_regularization_losses
trainable_variables
regularization_losses
layer_metrics
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
╦
_postprocessing_layers
	variables
trainable_variables
 regularization_losses
!	keras_api
V__call__
*W&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EncodingNetwork", "config": {"layer was saved without config": true}}
б

kernel
	bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"§
_tf_keras_layerс{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 6171, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
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
 "
trackable_list_wrapper
Г
&non_trainable_variables

'layers
(metrics
	variables
)layer_regularization_losses
trainable_variables
regularization_losses
*layer_metrics
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
-non_trainable_variables

.layers
/metrics
	variables
0layer_regularization_losses
trainable_variables
 regularization_losses
1layer_metrics
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
2non_trainable_variables

3layers
4metrics
"	variables
5layer_regularization_losses
#trainable_variables
$regularization_losses
6layer_metrics
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Р
7	variables
8trainable_variables
9regularization_losses
:	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"М
_tf_keras_layer╣{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
├

kernel
bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
\__call__
*]&call_and_return_all_conditional_losses"ъ
_tf_keras_layerё{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 10]}}
 "
trackable_list_wrapper
.
+0
,1"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
?non_trainable_variables

@layers
Ametrics
7	variables
Blayer_regularization_losses
8trainable_variables
9regularization_losses
Clayer_metrics
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Dnon_trainable_variables

Elayers
Fmetrics
;	variables
Glayer_regularization_losses
<trainable_variables
=regularization_losses
Hlayer_metrics
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
Ј2ї
*__inference_polymorphic_action_fn_23400688
*__inference_polymorphic_action_fn_23400743▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ж2Т
0__inference_polymorphic_distribution_fn_23400781▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
&__inference_get_initial_state_23400784д
Ю▓Ў
FullArgSpec!
argsџ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
░BГ
__inference_<lambda>_46102"ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
░BГ
__inference_<lambda>_46099"ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЗBы
&__inference_signature_wrapper_23400598
0/discount0/observation0/reward0/step_type"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
лB═
&__inference_signature_wrapper_23400610
batch_size"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┬B┐
&__inference_signature_wrapper_23400625"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┬B┐
&__inference_signature_wrapper_23400632"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
	J
Const9
__inference_<lambda>_46099б

б 
ф "і 	2
__inference_<lambda>_46102б

б 
ф "ф S
&__inference_get_initial_state_23400784)"б
б
і

batch_size 
ф "б в
*__inference_polymorphic_action_fn_23400688╝	^яб┌
мб╬
к▓┬
TimeStep,
	step_typeі
	step_type         &
rewardі
reward         *
discountі
discount         4
observation%і"
observation         

б 
ф "R▓O

PolicyStep&
actionі
action         
stateб 
infoб Њ
*__inference_polymorphic_action_fn_23400743С	^єбѓ
ЩбШ
Ь▓Ж
TimeStep6
	step_type)і&
time_step/step_type         0
reward&і#
time_step/reward         4
discount(і%
time_step/discount         >
observation/і,
time_step/observation         

б 
ф "R▓O

PolicyStep&
actionі
action         
stateб 
infoб ­
0__inference_polymorphic_distribution_fn_23400781╗	^яб┌
мб╬
к▓┬
TimeStep,
	step_typeі
	step_type         &
rewardі
reward         *
discountі
discount         4
observation%і"
observation         

б 
ф "л▓╠

PolicyStepб
actionЌњЊ­р├Ѓ}бz
`
Cб@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*ф'
%
locі
Identity         
ф _TFPTypeSpec
stateб 
infoб ║
&__inference_signature_wrapper_23400598Ј	^пбн
б 
╠ф╚
.

0/discount і

0/discount         
8
0/observation'і$
0/observation         

*
0/rewardі
0/reward         
0
0/step_type!і
0/step_type         "+ф(
&
actionі
action         a
&__inference_signature_wrapper_2340061070б-
б 
&ф#
!

batch_sizeі

batch_size "ф Z
&__inference_signature_wrapper_234006250б

б 
ф "ф

int64і
int64 	>
&__inference_signature_wrapper_23400632б

б 
ф "ф 
package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.Declaration;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.Objects;

@Data
public class VariableExpression extends Declaration {
    private Type type;
    private Name name;
    private Expression initializer;

    public VariableExpression(Node prarent, Type type, Name name) {
        super(prarent);
        this.type = type;
        this.name = name;

        this.type.setPrarent(this);
        this.name.setPrarent(this);

        getChildrens().addAll(type, name);
    }

    public VariableExpression(Node prarent, Type type, Name name, Expression initializer) {
        super(prarent);
        this.type = type;
        this.name = name;
        this.initializer = initializer;

        this.type.setPrarent(this);
        this.name.setPrarent(this);
        this.initializer.setPrarent(this);

        getChildrens().addAll(type, name, initializer);
    }

    public static void parser(Node node) {
        if (node instanceof VariableExpression) return;
        List<Node> nodes = node.getChildrens().stream().filter(a -> Field_Modifiers.contains(a.getTokenType())).toList();
        node.getChildrens().removeAll(nodes);

        Stream.of(node.getChildrens()).reduce((list, a, b, c) -> {
            if (a instanceof Name && b instanceof Name) {
                VariableExpression declare = new VariableExpression(node, Type.getType(a), (Name) b);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Type && b instanceof Name) {
                VariableExpression declare = new VariableExpression(node, (Type) a, (Name) b);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Name && b instanceof AssignExpression d) {
                VariableExpression declare = new VariableExpression(node, Type.getType(a), d.getTarget(), d.getValue());
                d.getValue().setPrarent(declare);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Type && b instanceof AssignExpression d) {
                VariableExpression declare = new VariableExpression(node, (Type) a, d.getTarget(), d.getValue());
                d.getValue().setPrarent(declare);
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            }
        });
    }

    @Override
    public String toString() {
        String concat = type.toString().concat(" ").concat(name.toString()).concat(" ");
        if (Objects.nonNull(initializer)) concat = concat.concat("= ").concat(initializer.toString()).concat(";");
        return concat;
    }
}
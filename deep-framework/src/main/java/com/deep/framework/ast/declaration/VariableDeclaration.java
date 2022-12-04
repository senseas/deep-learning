package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.AssignExpression;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ELLIPSIS;

@Data
public class VariableDeclaration extends Declaration {
    private Type type;
    private Name name;
    private Expression initializer;

    public VariableDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof VariableDeclaration) return;
        List<Node> nodes = node.getChildrens().stream().filter(a -> Field_Modifiers.contains(a.getTokenType())).toList();
        node.getChildrens().removeAll(nodes);

        Stream.of(node.getChildrens()).reduce((list, a, b, c) -> {
            if (a instanceof Name && b instanceof Name) {
                VariableDeclaration declare = new VariableDeclaration(node);
                Type type = Type.getType(a);
                declare.setType(type);
                declare.setName((Name) b);
                declare.setChildrens(Stream.of(declare.getType(), declare.getName()));
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Name && b instanceof AssignExpression d) {
                VariableDeclaration declare = new VariableDeclaration(node);
                d.getValue().setPrarent(declare);
                Type type = Type.getType(a);
                declare.setType(type);
                declare.setName(d.getTarget());
                declare.setInitializer(d.getValue());
                declare.setChildrens(Stream.of(declare.getType(), declare.getName(), declare.getInitializer()));
                node.replaceAndRemove(a, declare, b);
                list.remove(b);
            } else if (a instanceof Name && Objects.nonNull(b) && b.equals(ELLIPSIS) && c instanceof Name) {
                VariableDeclaration declare = new VariableDeclaration(node);
                Type type = Type.getType(a);
                declare.setType(type);
                declare.setName((Name) c);
                node.replace(a, declare);
                node.getChildrens().removeAll(List.of(b, c));
                list.removeAll(List.of(b, c));
            }
        });
    }

}